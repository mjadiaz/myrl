import torch
import os
import numpy as np
from modular.memory.base import Experience

class ModelIO:
    def __init__(self,  save_path, name):
        self._save_path = save_path
        self._name = name
        self._model_file = os.path.join(self._save_path, self._name)
    def save_model(self, state_dict):
        if not(os.path.exists(self._save_path)):
            os.makedirs(self._save_path)
        torch.save(state_dict, self._model_file)
    def load_model(self, model):
        model.load_state_dict(torch.load(self._model_file))
        
def experiences_to_numpy(experiences):
    '''
    Takes a batch of experiences: np.array[Experience] and transforms
    the experiences into separate np.arrays.

    Args:
    ----
    experiences: np.array[Experience, Experience, ...]
    
    Returns:
    -------
    states: np.array
    actions: np.array
    rewards: np.array
    new_states: np.array
    dones: np.array
    '''
    batch_size = len(experiences)
    states = np.array([experiences[i].state for i in range(batch_size)])
    actions = np.array([experiences[i].action for i in range(batch_size)])
    rewards = np.array([experiences[i].reward for i in range(batch_size)])
    new_states = np.array([experiences[i].new_state for i in range(batch_size)])
    dones = np.array([experiences[i].done for i in range(batch_size)])
    return (states, actions, rewards, new_states, dones)

def experiences_to_tensor(batch, device):
    '''
    Takes a the arrays of SARSD and transforms them 
    to torch.tensors.

    Args: 
    ----
    batch: Tuple[np.array, np.array, ...] SARSD

    Returns:
    -------
    states: torch.tensor
    actions: torch.tensor
    rewards: torch.tensor
    new_states: torch.tensor
    dones: torch.tensor
    device: torch device available
    '''
    states, actions, rewards, new_states, dones = batch
    states = torch.tensor(states).float().to(device) 
    actions = torch.tensor(actions).float().to(device)    
    rewards = torch.tensor(rewards).float().to(device)    
    new_states = torch.tensor(new_states).float().to(device)    
    dones = torch.tensor(dones).float().to(device) 
    return (states, actions, rewards, new_states, dones)
   

def random_experience(state_dimension: int, action_dimension: int):
    '''
    Create a random experience.

    Args:
    ----
    state_dimension
    action_dimension

    Returns:
    --------
    Experience(S,A,R,S,D)
    '''
    state = np.random.random(state_dimension)
    action = np.random.random(action_dimension)
    reward = np.random.random()
    new_state = np.random.random(state_dimension)
    done = True if np.random.random() > 0.5 else False
    experience = Experience(state, action, reward, new_state, done)
    return experience

def running_mean(data: np.ndarray, kernel_size: int = 10):
    '''
    Calculates the running average in a kernel_size window
    '''
    kernel = np.ones(kernel_size)/kernel_size
    data_convolved = np.convolve(data, kernel, mode='valid')
    return data_convolved
