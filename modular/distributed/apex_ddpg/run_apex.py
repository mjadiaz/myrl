from omegaconf import OmegaConf, DictConfig
import numpy as np

import torch

import gym
import datetime
import ray

from actor import Actor
from memory import ReplayBuffer
from learner import Learner
from parameter_server import ParameterServer
from utils import HyperParams, Writer 

def main():
    # Read hyper parameters
    hyper_parameters = HyperParams("agent_configs/apex_ddpg.yaml").get_config()
    
    # Ray actors init
    ray.init(local_mode=False)

    writer = Writer.remote("tests/apex_ddpg_1")
    parameter_server = ParameterServer.remote()
    replay_buffer = ReplayBuffer.remote()
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
                eps
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
                True
                )
        eval_actors_ids.append(actor)

    # Start collecting experiences and learning
   
    learner.start_learning.remote()
    
    total_samples = 0
    best_eval_mean_reward = np.NINF
    eval_mean_rewards = []
    
    while total_samples < hyper_parameters.agent.max_samples:
        tsid = replay_buffer.get_total_env_samples.remote()
        new_total_samples = ray.get(tsid)
        if (new_total_samples - total_samples
                >= hyper_parameters.agent.timesteps_per_iteration):
            total_samples = new_total_samples
            print("Total samples:", total_samples)
            parameter_server.set_eval_weights.remote()
            eval_sampling_ids = []
            for eval_actor in eval_actor_ids:
                sid = eval_actor.sample.remote()
                eval_sampling_ids.append(sid)
            eval_rewards = ray.get(eval_rewards)
            print("Evaluation rewards: {}".format(eval_rewards))
            eval_mean_reward = np.mean(eval_rewards)
            eval_mean_rewards.append(eval_mean_reward)
            print("Mean evaluation reward: {}".format(eval_mean_reward))
            writer.add_scalar(
                    "Mean evaluation reward",
                    eval_mean_reward,
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

if __name__ == '__main__':
    main()
