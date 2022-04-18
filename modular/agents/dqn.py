from omegaconf import OmegaConf, DictConfig
import numpy as np
from modular.memory.memory import PrioritizedReplayMemory, DequeReplay, Experience
from modular.common.utils import experiences_to_tensor
from modular.networks.fc_nets import DQNfc

import torch
import torch.nn.functional as F

import gym

class HyperParams:
    def __init__(self, agent_file: str):
        self._hp = OmegaConf.load(agent_file)
        self.alpha = self._hp.agent.alpha
        self.beta = self._hp.agent.beta
        self.gamma = self._hp.agent.gamma
        self.create_env() 
        self.create_network('dqn', self.alpha )
        self.create_network('target_dqn', self.alpha )
        self.create_memory()
        
    def __repr__(self):
        return str(self._hp)
    
    def get_config(self):
        return self._hp
    def create_env(self):
        env = gym.make(self._hp.env.name)
        config = OmegaConf.create(
                {'env': {
                    'state_dimension': env.observation_space.shape[0],
                    'action_dimension': env.action_space.n
                    }})
        self._hp = OmegaConf.merge(self._hp, config)
    def create_network(self, network: str, lr: float):
        '''
        Create a config for the network and merge 
        with agent config
        '''
        config = OmegaConf.create(
                {network: { 
                    'name': network, 
                    'save_path': self._hp.agent.save_path,
                    'learning_rate': lr,
                    'state_dimension': self._hp.env.state_dimension,
                    'action_dimension': self._hp.env.action_dimension}})
        self._hp = OmegaConf.merge(self._hp, config)
    def create_memory(self):
        config = OmegaConf.create({
            'memory':{
                'batch_size': self._hp.agent.batch_size,
                'max_size': self._hp.agent.max_size,
                'action_dimension': self._hp.env.action_dimension,
                'state_dimension': self._hp.env.state_dimension}})
        self._hp = OmegaConf.merge(self._hp, config)

 

class DQN:
    def __init__(self, hyper_params: DictConfig):
        self._hp = hyper_params

        # Parameters from the hyper_params: DictConfig
        self._gamma = self._hp.agent.gamma
        self._initial_exploration = self._hp.agent.initial_exploration
        self._final_exploration = self._hp.agent.final_exploration
        self._final_exploration_frame = self._hp.agent.final_exploration_frame
        self._tau = self._hp.agent.tau
        self._state_dimenion = self._hp.env.state_dimension
        self._action_dimension = self._hp.env.action_dimension
        self._max_size = self._hp.agent.max_size
        self._batch_size = self._hp.agent.batch_size
        self._save_path = self._hp.agent.save_path
        self.double_dqn = self._hp.agent.double_dqn
        
        # Create Networks
        self.dqn = DQNfc(self._hp.dqn)
        self.target_dqn = DQNfc(self._hp.dqn)
        self.update_target_networks(tau=1.)

        # Initialize memory
        self._per = self._hp.memory.prioritized
        if self._per:
            self.memory = PrioritizedReplayMemory(self._hp.memory)
        else:
            self.memory = DequeReplay(self._hp.memory)

        # Device
        self.device = self.dqn.device
        # Exploration
        self._frame = 0
        self._epsilon = self._initial_exploration
    def update_target_networks(self, tau: float = None):
        '''
        Update the target network parameters. 
        The target network can also slowly track the original network 
        by modifiying the tau parameter.

        Args:
        -----
        tau: float
        '''
        if tau == None:
            tau = self._tau
        def parameters_update(network, target_network, tau=tau):
            net_params = dict(network.named_parameters())
            target_net_params = dict(target_network.named_parameters())

            for name in net_params:  
                net_params[name] = tau*net_params[name].clone() \
                        + (1. - tau)*target_net_params[name].clone()

            target_network.load_state_dict(net_params)

        parameters_update(self.dqn, self.target_dqn)
    def update_exploration(self):
        if self._frame < self._final_exploration_frame:
            delta_epsilon = abs(self._initial_exploration - self._final_exploration)/self._final_exploration_frame
            self._epsilon =  self._epsilon - delta_epsilon
        else:
            self._epsilon = self._final_exploration

    @torch.no_grad()
    def select_action(self, state: np.ndarray):
        '''
        Forwards the state through the DQN model to get
        the action prediction
        
        Args:
        ----
        state: np.ndarray

        Returns:
        -------
        max_q_vale: if max_q_value exist. None otherwise.
        action: int
        '''
        self._frame += 1
        self.update_exploration()

        if np.random.random() < self._epsilon:
            action = np.random.choice(np.arange(self._action_dimension))
            max_q_value = None
            return max_q_value, action
        else:
            self.dqn.eval()
            state = torch.tensor(state).float().to(self.device)
            q_values = self.dqn(state)
            max_q_value, action = torch.max(q_values.cpu(), dim=0)
            action = int(action.item())
            self.dqn.train()
            return max_q_value, action
    
    def remember(self, state, action, reward, new_state, done):
        experience = Experience(state, action, reward, new_state, done)
        self.memory.add(experience)

    def enough_experience(self):
        if self.memory.__len__() > self._batch_size:
            return True
        else:
            return False

    def learn(self):
        if self._per:
            self._learn_with_per()
        else:
            self._learn_with_uniform_memory()

    def _learn_with_uniform_memory(self):
        if not(self.enough_experience()):
            return 

        experience_batch = self.memory.sample()
        
        states, actions, rewards, new_states, dones =\
                experiences_to_tensor(experience_batch, self.device)
        
        self.target_dqn.eval()
        self.dqn.eval()
        if self.double_dqn:
            q_values = self.dqn(states)
            max_q_values = torch.gather(
                    q_values, 1, actions.unsqueeze(-1).to(torch.int64)) 
            max_q_values = max_q_values.flatten()
            astar = torch.argmax(q_values, dim=1)
            qs = self.target_dqn(new_states).gather(
                    dim=1, index=astar.unsqueeze(dim=1)).squeeze()
            
            y = rewards + self._gamma * qs.detach() *\
                    (torch.ones(self._batch_size).to(self.device) - dones)
        else:

            target_q_values = self.target_dqn(new_states)
            target_q_max, target_actions = torch.max(target_q_values, dim=1)
            
            q_values = self.dqn(states)
            max_q_values = torch.gather(
                    q_values, 1, actions.unsqueeze(-1).to(torch.int64)) 
            max_q_values = max_q_values.flatten()
            #max_q_values, _ = torch.max(q_values,dim=1)
            
            y = rewards + self._gamma * target_q_max *\
                    (torch.ones(self._batch_size).to(self.device) - dones)
        
        self.dqn.train()
        self.dqn.optimizer.zero_grad()
    
        dqn_loss = F.mse_loss(y, max_q_values)
        dqn_loss.backward()
        self.dqn.optimizer.step()

        if (self._frame % 100 == 0):
            self.update_target_networks()


    def save_models(self):
        self.dqn.save_model()
        self.target_dqn.save_model()

    def load_models(self):
        self.dqn.load_model()
        self.target_dqn.load_model()


        
        

