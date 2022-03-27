from omegaconf import OmegaConf, DictConfig
import numpy as np
import gym
import torch
import os

from modular.networks.fc_nets import DDPGActor, DDPGCritic
from modulasr.exploration.noises import OUActionNoise
from modular.common.utils import experiences_to_tensor

import ray

class HyperParamsActor:
        def __init__(self, agent_file: str, env_config: DictConfig = None):
                self.hp = OmegaConfig.load(agent_file)
                self.alpha = self.hp.agent.alpha
                self.beta  = self.hp.agent.beta
                self.gamma = self.hp.agent.gamma
                self.env_config = env_config

                self.create_env()
                self.create_network('actor', self.alpha)
        def __repr__(self):
                return str(self.hp)
        def get_config(self):
                return self.hp
        def create_env(self):
                if not(self.env_config == None):
                        env = gym.make(self.hp.env.name, env_config = self.env_config)
                else:
                        env = gym.make(self.hp.env.name)

@ray.remote
class Actor:
        def __init__(self, hyper_params: DictConfig):
                self.hp = hyper_params
                self.alpha = self.hp.agent.alpha
                #self.beta = self.hp.agent.beta
                #self.tau = self.hp.agent.tau
                self.gamma = self.hp.agent.gamma
                self.state_dimension = self.hp.env.state_dimension
                self.action_dimension = self.hp.env.action_dimension

                # Internal Experience replay for memory efficiency
                #self.actor_max_size = self.hp.agent.actor_max_size

                # Create Networks
                self.actor = DDPGActor(self.hp.actor)
                self.actor.device = 'cpu'

                # Initialize noise
                self.noise = OUActionNoise(mu = np.zeros(self.action_dimension))

                # Sync parameters with parameter server
                self.pull_parameters()

                self.device = self.actor.device


