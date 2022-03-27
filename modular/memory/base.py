'''
Adapted from: 
    
- https://github.com/cyoon1729/RLcycle/blob/master/rlcycle/common/abstract/buffer.py
- https://davidrpugh.github.io/stochastic-expatriate-descent/pytorch/deep-reinforcement-learning/deep-q-networks/2020/04/14/prioritized-experience-replay.html
'''

import numpy as np

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Tuple
from omegaconf import DictConfig
from collections import namedtuple

#@dataclass(frozen=True)
#class Experience:
#    state: np.ndarray
#    action: np.ndarray
#    reward: any 
#    new_state: np.ndarray
#    done: bool
Experience = namedtuple(
    'Experience',
    field_names=[
        'state',
        'action',
        'reward',
        'new_state',
        'done'
        ]
    )

class ReplayMemoryBase(ABC):
    '''
    Abstract base class for replay memory
    '''
    
    @property
    @abstractmethod
    def memory(self):
        ''' Buffer usually implemented by collections.deque'''
        pass

    @property
    @abstractmethod
    def hyper_params(self):
        '''
        DictConfig containing hyper parameters like:
        - max_size
        - batch_size
        '''
        pass
   
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def add(self, experience: Experience):
        ''' Add method for adding an experience to the memory '''
        pass

    @abstractmethod
    def sample(self):
        ''' 
        Sample method for sampling from memory 
        '''
        pass

