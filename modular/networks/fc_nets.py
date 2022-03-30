from modular.networks.base import NeuralNetBase 
from modular.common.utils import ModelIO

import torch
import torch.nn as nn
import torch.optim as optim
import os

from omegaconf import OmegaConf, DictConfig

       

class DDPGActor(nn.Module, NeuralNetBase):
    '''
    Actor for DDPG algorithm. 

    Args
    ----
    hyper_params: DictConfig:
        - name: Name for the Net, usually Actor.
        - save_path: Path for saving the model checkpoint.
        - learning_rate
        - state_dimension: Dimension of the environment state
        - action_dimension: Dimension of the environment action
    '''
    def __init__(self, hyper_params: DictConfig):
        super().__init__()
        self._hyper_params = hyper_params
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Hyper parameteres derived from the hyper_params: DictConfig
        self._name = self._hyper_params.name
        self._save_path = self._hyper_params.save_path
        self._learning_rate = self._hyper_params.learning_rate
        self._state_dimension = self._hyper_params.state_dimension
        self._action_dimension = self._hyper_params.action_dimension

        self._model_io = ModelIO(self._save_path, self._name)

        
        self._linear_block = nn.Sequential(
                nn.Linear(self._state_dimension, 400),
                nn.LayerNorm(400),
#                nn.BatchNorm1d(400, track_running_stats=False),
                nn.ReLU(),
                nn.Linear(400,300),
                nn.LayerNorm(300),
                #nn.BatchNorm1d(300, track_running_stats=False),
                nn.ReLU(),
                nn.Linear(300, self._action_dimension),
                nn.Tanh()
                )
        self.optimizer = optim.Adam(self.parameters(), lr=self._learning_rate) 
        self.to(self.device)

    @property
    def hyper_params(self):
        return self._hyper_params

    @property
    def device(self):
        return self._device
    @device.setter
    def device(self, value):
        self._device =  value

    def forward(self, state):
        a = self._linear_block(state)
        return a
    
    def save_model(self):
        self._model_io.save_model(self.state_dict())

    def load_model(self):
        self._model_io.load_model(self)

class DDPGCritic(nn.Module, NeuralNetBase):
    '''
    Actor for DDPG algorithm. 

    Args
    ----
    hyper_params: DictConfig:
        - name: Name for the Net, usually Critic. 
        - save_path: Path for saving the model checkpoint.
        - learning_rate
        - state_dimension: Dimension of the environment state
        - action_dimension: Dimension of the environment action
    '''
    def __init__(self, hyper_params: DictConfig):
        super().__init__()
        self._hyper_params = hyper_params
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Hyper parameteres derived from the hyper_params: DictConfig
        self._name = self._hyper_params.name
        self._save_path = self._hyper_params.save_path
        self._learning_rate = self._hyper_params.learning_rate
        self._state_dimension = self._hyper_params.state_dimension
        self._action_dimension = self._hyper_params.action_dimension

        self._model_io = ModelIO(self._save_path, self._name)

        self._linear_block = nn.Sequential(
                nn.Linear(self._state_dimension + self._action_dimension, 400),
#                nn.BatchNorm1d(400,track_running_stats=False),
                nn.LayerNorm(400),
                nn.ReLU(),
                nn.Linear(400,300),
                nn.LayerNorm(300),
                #nn.BatchNorm1d(300, track_running_stats=False),
                nn.ReLU(),
                nn.Linear(300,1)
                )


        self.optimizer = optim.Adam(self.parameters(), lr=self._learning_rate) 
        self.to(self.device)

    @property
    def hyper_params(self):
        return self._hyper_params

    @property
    def device(self):
        return self._device
    
    def forward(self, state, action):
        if not(len(state.shape) == 1):
            q = self._linear_block(torch.cat([state, action], 1))
            return q
        else:
            q = self._linear_block(torch.cat([state, action], 0))
            return q


    def save_model(self):
        self._model_io.save_model(self.state_dict())

    def load_model(self):
        self._model_io.load_model(self)
        

class DQNfc(nn.Module, NeuralNetBase):
    '''

    '''
    def __init__(self, hyper_params: DictConfig):
        super().__init__()
        self._hyper_params = hyper_params
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Hyper parameteres derived from the hyper_params: DictConfig
        self._name = self._hyper_params.name
        self._save_path = self._hyper_params.save_path
        self._learning_rate = self._hyper_params.learning_rate
        self._state_dimension = self._hyper_params.state_dimension
        self._action_dimension = self._hyper_params.action_dimension
        self._model_io = ModelIO(self._save_path, self._name)

        self.linear_block = nn.Sequential(
                nn.Linear(self._state_dimension, 128),
                nn.ReLU(),
                nn.Linear(128, self._action_dimension) 
                )
        self.optimizer = optim.Adam(self.parameters(), lr=self._learning_rate) 
        self.to(self.device)


    def forward(self, x):
        x = self.linear_block(x)
        return x
    @property
    def hyper_params(self):
     return self._hyper_params
    @property
    def device(self):
        return self._device

    def save_model(self):
        self._model_io.save_model(self.state_dict())

    def load_model(self):
        self._model_io.load_model(self)
