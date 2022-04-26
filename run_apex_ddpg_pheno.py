from omegaconf import OmegaConf, DictConfig
import numpy as np

import torch

import gym
import datetime
import ray
import argparse

from modular.distributed.apex_ddpg.actor import Actor
from modular.distributed.apex_ddpg.memory import ReplayBuffer
from modular.distributed.apex_ddpg.learner import Learner
from modular.distributed.apex_ddpg.parameter_server import ParameterServer
from modular.distributed.apex_ddpg.utils import HyperParams, Writer 

from modular.networks.fc_nets import DDPGActor, DDPGCritic

from pheno_game.envs.pheno_env import PhenoEnvContinuous_v0

def train(save_path):
    # Read hyper parameters
    env_config = OmegaConf.load("hep_tools.yaml")
    #env_config = None
    hyper_parameters = HyperParams(
        "agent_configs/apex_ddpg.yaml",
        env_config=env_config,
        save_path=save_path,
        ).get_config()
    print(hyper_parameters.agent.save_path)
    # Ray actors init
    ray.init(local_mode=False)

    writer = Writer(hyper_parameters.agent.save_path)
    parameter_server = ParameterServer.remote(
            DDPGActor(hyper_parameters.actor),
            DDPGCritic(hyper_parameters.critic)
            )
    replay_buffer = ReplayBuffer.remote(
            hyper_parameters
            )
    learner = Learner.remote(
            hyper_parameters,
            replay_buffer,
            parameter_server
            )
    
    training_actors_ids = []
    eval_actors_ids = []

    for i in range(hyper_parameters.agent.num_workers):
        eps = hyper_parameters.agent.max_exploration_eps\
                * i / hyper_parameters.agent.num_workers
        actor = Actor.remote(
                "train-"+str(i),
                replay_buffer,
                parameter_server,
                hyper_parameters,
                eps,
                env_config=env_config
                )
        actor.sample.remote()
        training_actors_ids.append(actor)
    for i in range(hyper_parameters.agent.num_workers):
        eps = 0
        actor = Actor.remote(
                "eval-" + str(i),
                replay_buffer,
                parameter_server,
                hyper_parameters,
                eps,
                True,
                env_config=env_config
                )
        eval_actors_ids.append(actor)

    # Start collecting experiences and learning
   
    learner.start_learning.remote()
    
    total_samples = 0
    best_eval_mean_reward = np.NINF
    eval_mean_rewards = []
    eval_mean_lens = []
    
    while total_samples < hyper_parameters.agent.max_samples:
        tsid = replay_buffer.get_total_env_samples.remote()
        new_total_samples = ray.get(tsid)
        if (new_total_samples - total_samples
                >= hyper_parameters.agent.timesteps_per_iteration):
            total_samples = new_total_samples
            print("Total samples:", total_samples)
            parameter_server.set_eval_weights.remote()
            eval_sampling_ids = []
            for eval_actor in eval_actors_ids:
                sid = eval_actor.sample.remote()
                eval_sampling_ids.append(sid)
            eval_rewards_lens = ray.get(eval_sampling_ids)
            eval_rewards_lens = np.array(eval_rewards_lens)
            eval_rewards = eval_rewards_lens[:,0]
            eval_lens = eval_rewards_lens[:,1]
            print("Evaluation rewards: {}".format(eval_rewards))
            print("Evaluation lens: {}".format(eval_lens))
            eval_mean_reward = np.mean(eval_rewards)
            eval_mean_rewards.append(eval_mean_reward)
            eval_mean_len = np.mean(eval_lens)
            eval_mean_lens.append(eval_mean_len)
            print("Mean evaluation reward: {}".format(eval_mean_reward))
            print("Mean evaluation length: {}".format(eval_mean_len))
            writer.add_scalar(
                    "Mean evaluation reward",
                    eval_mean_reward,
                    total_samples
                    )
            writer.add_scalar(
                    "Mean evaluation length",
                    eval_mean_len,
                    total_samples
                    )
            if eval_mean_reward > best_eval_mean_reward:
                print("Model has improved! Saving the model")
                best_eval_mean_reward = eval_mean_reward
                parameter_server.save_eval_weights.remote()

    print("Finishing the training.")
    for actor in training_actors_ids:
        actor.stop.remote()
    learner.stop.remote()

def test(render, n_episodes):
    def select_action(actor, state, hp):
        state = torch.tensor(state).float().to(actor.device)
        action = actor(state)
        action = np.clip(
            action.cpu().detach().numpy(),
            hp.env.action_min,
            hp.env.action_max
            )
        return action

    hyper_parameters = HyperParams("agent_configs/apex_ddpg.yaml").get_config()

    actor = DDPGActor(hyper_parameters.actor_server)

    actor.load_model()

    env = gym.make(hyper_parameters.env.name)

    for i in range(n_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        while not done:
            action = select_action(actor, state, hyper_parameters)
            new_state, reward, done, info = env.step(action)
            #self.remember(state, action, reward, new_state, done)
            state = new_state
            episode_reward += reward
            episode_length += 1
            if render:
                env.render()
        print("Episode reward:", episode_reward)
        print("Episode Length:", episode_length)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--train',
            type=bool,
            help = 'Train Apex DDPG'
            )
    parser.add_argument(
            '-sp',
            '--save_path',
            type=str,
            help='Path and name for the training run. example: tests/apex'
            )
    parser.add_argument(
            '--test',
            type=bool,
            help = "Test the saved model."
            )
    parser.add_argument(
            '-r',
            "--render", 
            type=bool ,
            help="Test with render. True or False"
            )
    parser.add_argument(
            '-n',
            '--n_episodes',
            type=int,
            help="Number of episodes to test."
            )
    args = parser.parse_args()
    
    if args.test:
        test(args.render, args.n_episodes)
    if args.train:
        train(save_path=args.save_path)

if __name__ == '__main__':
    main()
