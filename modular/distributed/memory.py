import ray
import numpy as np
import gym

from modular.memory.memory import DequeReplay, Experience
from modular.common.utils import experiences_to_numpy, experiences_to_tensor
from omegaconf import DictConfig

@ray.remote
class GlobalMemory:
    def __init__(self, hyper_params: DictConfig):
        self.hp = hyper_params
        self.max_size = self.hp.max_size
        self.memory = DequeReplay(self.hp)
        self.step_counter = 0
        self.batch_size = self.hp.batch_size
        
    def add(self, experience):
        self.memory.add(experience)
        self.increment_step()
        #print(f'step_counter: {self.step_counter}')
    def get_memory(self):
        return self.memory.memory
    
    def increment_step(self):
        self.step_counter += 1

    def get_step_counter(self):
        return self.step_counter
    
    def is_full(self):
        if self.get_step_counter() >= self.max_size:
            return True
        else:
            return False
    def able_to_learn(self):
        #print('memory: able to learn?')
        if self.step_counter >= self.batch_size:
            #print('memory: True')
            return True
        else:
            #print('memory: False')
            return False

    
    def sample(self):
        return  self.memory.sample()
