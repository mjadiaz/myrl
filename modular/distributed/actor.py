from omegaconf import OmegaConf, DictConfig
import numpy as np
import gym
import torch
import os

from modular.networks.fc_nets import DDPGActor, DDPGCritic
from modular.exploration.noises import OUActionNoise
from modular.common.utils import experiences_to_tensor
from modular.memory.memory import DequeReplay, Experience
from pheno_game.envs.pheno_env import PhenoEnvContinuous_v0

import ray

class HyperParamsActor:
    def __init__(self, agent_file: str, env_config: DictConfig = None):
        self.hp = OmegaConf.load(agent_file)
        self.alpha = self.hp.agent.alpha
        self.beta  = self.hp.agent.beta
        self.gamma = self.hp.agent.gamma
        self.env_config = env_config
    
        self.create_env()
        self.create_network('actor', self.alpha)
        self.create_network('target_actor', self.alpha)
        self.create_network('critic', self.beta)
        self.create_network('target_critic', self.beta)
        self.create_memory()

        self.hp.env_config = self.env_config
    def __repr__(self):
        return str(self.hp)
    def get_config(self):
        return self.hp
    def create_env(self):
        if not(self.env_config == None):
            #env = gym.make(self.hp.env.name, env_config = self.env_config)
            env = PhenoEnvContinuous_v0(env_config=self.env_config)
        else:
            env = gym.make(self.hp.env.name)
        config = OmegaConf.create(
            {"env": {
                'state_dimension': env.observation_space.shape[0],
                'action_dimension': env.action_space.shape[0],
                'action_min': env.action_space.low.tolist(),
                'action_max': env.action_space.high.tolist(),
                }}
        )
        self.hp = OmegaConf.merge(self.hp, config)
    def create_network(self, network: str, lr: float):
        config = OmegaConf.create(
            {network:{
                'name': network,
                'save_path': self.hp.agent.save_path,
                'learning_rate': lr,
                'state_dimension': self.hp.env.state_dimension,
                'action_dimension': self.hp.env.action_dimension,
            }}
        )
        self.hp = OmegaConf.merge(self.hp, config)
    def create_memory(self):
        config = OmegaConf.create(
            {'memory':{
                'batch_size': self.hp.agent.batch_size,
                'max_size': self.hp.agent.max_size,
                'action_dimension': self.hp.env.action_dimension,
                'state_dimension': self.hp.env.state_dimension,
                }
             }
        )
        self.hp = OmegaConf.merge(self.hp, config)

@ray.remote
class Actor:
    def __init__(
            self,
            actor_id,
            parameter_server,
            global_memory,
            hyper_params: DictConfig
        ):
        self.hp = hyper_params
        self.alpha = self.hp.agent.alpha
        #self.beta = self.hp.agent.beta
        #self.tau = self.hp.agent.tau
        self.gamma = self.hp.agent.gamma
        self.state_dimension = self.hp.env.state_dimension
        self.action_dimension = self.hp.env.action_dimension
        self.actor_buffer_size = self.hp.agent.actor_buffer_size
        # Internal Experience replay for memory efficiency
        #self.actor_max_size = self.hp.agent.actor_max_size

        # Global Experience Replay Memory
        self.global_memory = global_memory
        
        # Parameter Server
        self.parameter_server = parameter_server
        self.updates_number_tracker = 0
        # Create Actor network in eval mode
        self.actor = DDPGActor(self.hp.actor)
        self.actor.device = 'cpu'
        self.actor.eval()

        # Initialize noise
        self.noise = OUActionNoise(mu = np.zeros(self.action_dimension))

        # Sync parameters with parameter server
        self.pull_parameters()
        self.device = self.actor.device

        #Initialize memory
        self.memory = DequeReplay(self.hp.memory)

        #Create Environment
        if not(self.hp.env_config == None):
            #self.env = gym.make(self.hp.env.name, env_config = self.hp.env_config)
            self.env = PhenoEnvContinuous_v0(env_config=self.hp.env_config)
            
        else:
            self.env = gym.make(self.hp.env.name)
        

    def pull_parameters(self):
        print('actor: pulling parameters from parameter server')
        weights = ray.get(self.parameter_server.get_weights.remote())
        self.actor.load_state_dict(weights)
        print('actor: weights updated from parameter server')

    def push_experience(self, experience):
        #print('actor: pushing experience')
        self.global_memory.add.remote(experience)

    def select_action(self, state: np.ndarray):
        '''
            Forwards the state though the actor network
        to get an action prediction.
        - The action is clipped 
        '''
        state = torch.tensor(state).float().to(self.device)
        action = self.actor(state) +\
            torch.tensor(self.noise()).float().to(self.device)
        action = np.clip(
            action.cpu().detach().numpy(),
            self.hp.env.action_min,
            self.hp.env.action_max
        )
        return action

    def remember(self, state, action, reward, new_state, done):
        experience = Experience(state, action, reward, new_state, done)
        #self.memory.add(experience)
        self.push_experience(experience)

    def enough_experience(self):
        if ray.get(self.global_memory.is_full.remote()):
            print('actor: is full')
            return True
        else:
            return False
        #if self.memory.__len__() >= self.actor_buffer_size:
        #    return True
        #else:
        #    return False
    def track_updates(self):
        updates = ray.get(self.parameter_server.get_updates_counter.remote())
        print('updates:', updates)
        if self.updates_number_tracker != updates:
            self.pull_parameters()
        self.updates_number_tracker = updates

    def run(self):
        while not self.enough_experience():
            self.noise.reset()
            state = self.env.reset()
            #print('actor:', state)
            done = False
            episode_reward = 0
            
            while not done:
                action = self.select_action(state)
                new_state, reward, done, info = self.env.step(action)
                self.remember(state, action, reward, new_state, done)
                state = new_state
                episode_reward += reward
                self.track_updates()
                #print('reward:', reward)
            #self.env.close()
            print(f'episode reward: {episode_reward}')


