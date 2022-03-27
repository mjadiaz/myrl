import numpy as np
from modular.memory.base import Experience, ReplayMemoryBase
from modular.common.utils import experiences_to_numpy, experiences_to_tensor
from omegaconf import DictConfig

from typing import Tuple

from collections import deque
from collections import namedtuple

class DequeReplay(ReplayMemoryBase):
    def __init__(self, hyper_params: DictConfig):
        self._hp = hyper_params
        self._batch_size = self._hp.batch_size
        self._max_size = self._hp.max_size
        self._memory = deque(maxlen=self._max_size)

    
    @property
    def hyper_params(self):
        return self._hp
    
    
    @property
    def memory(self):
        return self._memory

    def __len__(self):
        return len(self.memory)

    def add(self, experience: Experience):
        #exp = namedtuple('Experience', 
        #        field_names=[   'state', 'action', 'reward',
        #                        'new_state', 'done'])
        _exp = Experience(experience.state, experience.action, experience.reward, experience.new_state, experience.done)
        self.memory.append(_exp)
    def sample(self):
        indices = np.random.choice(self.__len__(), self._batch_size, replace=False)
        states, actions, rewards, new_states, dones =\
                zip(*[self.memory[idx] for idx in indices])
        
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        new_states = np.array(new_states)
        dones = np.array(dones)
        return (states, actions, rewards, new_states, dones)


    
class ExperienceReplayMemory(ReplayMemoryBase):
    ''' 
    Vanilla Experience Replay memory for storing past experiences
    
    Args:
    - hyper_params: DictConfig 
        With the parameters: 
        - batch_size: int = Maximum size of the memory
        - max_size: int = Size of the training batches
        - action_dimension = Dimension for the actions in the environment.
        - state_dimension = Dimension for the states in the environments.
        - warm_start: bool = Warm start required.
        - n_random_actions: int = number of random action for warm start.
        Note: Last 2 are for warm start.
    '''
    def __init__(self, hyper_params: DictConfig):
        self._hyper_params = hyper_params
        self._batch_size = self._hyper_params.batch_size
        self._max_size = self._hyper_params.max_size
        self._action_dimension = self._hyper_params.action_dimension
        self._state_dimension = self._hyper_params.state_dimension
        self._warm_start = self._hyper_params.warm_start
        self._memory = np.empty(self._max_size, dtype=Experience)
        self._memory_length = 0  #  Counting stored experiences
        self._memory_counter = 0  # Restart the counter when is reach to max size
        self._is_full = False
        if self._warm_start: 
            self._n_random_actions = self._hyper_params.n_random_actions
            self.warm_start(self._n_random_actions) 

    @property
    def memory(self):
        return self._memory
    @property
    def hyper_params(self):
        return self._hyper_params

    def __len__(self):
        return self._memory_length

    def add(self, experience: Experience):
        '''Add an experience to the memory'''
        self._memory[self._memory_counter] = experience
        if self.is_full() == False:
            self._memory_length += 1
        self._memory_counter += 1
        if self.max_counter():
            self._memory_counter = 0

    def is_full(self) -> bool:
        '''
        True if the number of added experiences is equal or 
        greater than max size, False otherwise
        '''
        return True if self._memory_length  == self._max_size else False
    def max_counter(self) -> bool:
        '''True if memory counter is equal to maximum size, False otherwise'''
        return True if self._memory_counter  == self._max_size else False

    def warm_start(self, n_random_actions: int):
        for _ in range(n_random_actions):
            self.add(random_experience(self.ac))

    def _sample(self) -> Tuple:
        if self._memory_length < self._max_size:
            experiences = self.memory[range(self._memory_length)]
        indices = np.random.choice( self._memory_length, self._batch_size,
                                    replace=False)
        experiences = np.random.permutation(experiences)
        experiences = self._memory[indices]
        return experiences
    
    def sample(self):
        '''
        Sample a batch of experiences from memory.
        
        Return:
        -------
        Tuple:
            states: np.array
            actions: np.array
            rewards: np.array
            new_states: np.array
            dones: np.array
        '''
        experiences = self._sample()
        states = np.array([experiences[i].state for i in range(self._batch_size)])
        actions = np.array([experiences[i].action for i in range(self._batch_size)])
        rewards = np.array([experiences[i].reward for i in range(self._batch_size)])
        new_states = np.array([experiences[i].new_state for i in range(self._batch_size)])
        dones = np.array([experiences[i].done for i in range(self._batch_size)])
        return (states, actions, rewards, new_states, dones)
        
class PrioritizedReplayMemory(ReplayMemoryBase):
    def __init__(self, hyper_params: DictConfig):
        self._hyper_params = hyper_params
        self._batch_size = self._hyper_params.batch_size
        self._max_size = self._hyper_params.max_size
        self._alpha = self._hyper_params.alpha
        self._beta = self._hyper_params.beta
        self._memory =  np.empty(   self._max_size, 
                                    dtype=[ ('priority', np.float32),
                                            ('experience', Experience)]
                                )
        self._memory_length = 0 #  Counting stored experiences
        random_state = self._hyper_params.random_state  #  ramdom_state: np.random.RandomState = None
        self._random_state = np.random.RandomState() if random_state is None else random_state


    def __len__(self) -> int:
        ''' Current number of prioritized experieces stored in the memory'''
        return self._memory_length
    
    @property
    def alpha(self):
        ''' Strength of prioritized sampling '''
        return self._alpha

    @property
    def memory(self):
        return self._memory
    
    @property
    def hyper_params(self):
        return self._hyper_params

    def add(self, experience: Experience):
        '''
        Add  a prioritized experience to the memory
        '''
        priority = 1. if self.is_empty() else self._memory['priority'].max()
        if self.is_full():
            if priority > self._memory['priority'].min():
                idx = self._memory['priority'].argmin()
                self._memory[idx] = (priority, experience)
            else:
                # Low priority experiences should not 
                # be included in the memory
                pass
        else:
            self._memory[self._memory_length] = (priority, experience)
            self._memory_length += 1

    def is_empty(self) -> bool:
        ''' True if the memory is empty, False otherwise '''
        return self._memory_length == 0

    def is_full(self) -> bool:
        ''' True if the memory is full, False otherwise '''
        return self._memory_length == self._max_size

    def _sample(self) -> Tuple:
        ''' Sample a batch of experiences from the memory'''
        # Use sampling scheme to determine which experiences to use for learning
        priorities = self._memory[:self._memory_length]['priority']
        sampling_probs = (priorities**self._alpha)/np.sum(priorities**self._alpha)
        idxs = self._random_state.choice(
                np.arange(priorities.size),
                size=self._batch_size,
                replace=True,
                p=sampling_probs
                )
        # Select the experiences and compute sampling
        experiences = self._memory['experience'][idxs]
        weights = (self._memory_length * sampling_probs[idxs])**-self._beta
        normalized_weights = weights / weights.max()

        return idxs, experiences, normalized_weights

    def sample(self):
        '''
        Sample from memory

        Return:
            - Tuple: idxs, states, actions, rewards, new_states, dones, normalized_weights
        '''
        idxs, experiences, normalized_weights = self._sample()
        experiences = experiences_to_numpy(experiences)
        return idxs,  experiences, normalized_weights

    def update_priorities(self, idxs: np.array, priorities: np.array) -> None:
        ''' Update the priorities associated with particular experiences '''
        self._memory['priority'][idxs] = priorities

   
