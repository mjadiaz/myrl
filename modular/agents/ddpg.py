from omegaconf import OmegaConf, DictConfig
import numpy as np
from modular.networks.fc_nets import DDPGActor, DDPGCritic
from modular.exploration.noises import OUActionNoise
from modular.memory.memory import PrioritizedReplayMemory, Experience, DequeReplay
from modular.common.utils import experiences_to_tensor
import torch
import torch.nn.functional as F
import pickle
import os
import gym
class HyperParamsDDPG:
    def __init__(self, agent_file: str, env_config: DictConfig = None):
        self._hp = OmegaConf.load(agent_file)
        self.alpha = self._hp.agent.alpha
        self.beta = self._hp.agent.beta
        self.gamma = self._hp.agent.gamma
        self._env_config = env_config
        self.create_env() 
        self.create_network('actor', self.alpha )
        self.create_network('target_actor', self.alpha )
        self.create_network('critic', self.beta )
        self.create_network('target_critic', self.beta )
        self.create_memory()
        
    def __repr__(self):
        return str(self._hp)
    
    def get_config(self):
        return self._hp
    def create_env(self):
        if not(self._env_config == None):
            env = gym.make(self._hp.env.name, env_config= self._env_config)
        else:
            env = gym.make(self._hp.env.name)
        config = OmegaConf.create(
                {'env': {
                    'state_dimension': env.observation_space.shape[0],
                    'action_dimension': env.action_space.shape[0],
                    'action_min': env.action_space.low.tolist(),
                    'action_max': env.action_space.high.tolist()}})
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

class DDPG:
    def __init__(self, hyper_params: DictConfig):
        self._hp = hyper_params

        # Parameters from the hyper_params: DictConfig
        self._alpha = self._hp.agent.alpha
        self._beta = self._hp.agent.beta
        self._tau = self._hp.agent.tau
        self._gamma = self._hp.agent.gamma
        self._state_dimension = self._hp.env.state_dimension
        self._action_dimension = self._hp.env.action_dimension
        self._max_size = self._hp.agent.max_size
        self._batch_size = self._hp.agent.batch_size
        self._save_path = self._hp.agent.save_path
        # Create Networks
        self.actor = DDPGActor(self._hp.actor)
        self.target_actor = DDPGActor(self._hp.target_actor)
        self.critic = DDPGCritic(self._hp.critic)
        self.target_critic = DDPGCritic(self._hp.target_critic)
        # Intialize noise
        self.noise = OUActionNoise(mu=np.zeros(self._action_dimension))
        # Copy networks parameters to target networks (tau =1 )
        self.update_target_networks(tau=1.)
        # Initialize Memory
        self._per = self._hp.memory.prioritized
        if self._per:
            self.memory = PrioritizedReplayMemory(self._hp.memory)
        else:
            #self.memory = ExperienceReplayMemory(self._hp.memory)
            self.memory = DequeReplay(self._hp.memory)
        # Save device
        self.device = self.actor.device

    def update_target_networks(self, tau: float = None):
        '''
        Update the parameters of the target networks slowly tracking 
        the principal networks

        Args:
        ----
        tau: Slow tracking parameter << 1
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

        parameters_update(self.actor, self.target_actor)
        parameters_update(self.critic, self.target_critic)

    def select_action(self, state: np.ndarray):
        '''
        Forwards the state through the actor network
        to get an action prediction.
        - The action is clipped

        Args:
        ----
        state: np.ndarray
        
        Return:
        ------
        action: np.array clipped action.
        '''
        self.actor.eval() 
        state = torch.tensor(state).float().to(self.device)
        action = self.actor(state) + \
                torch.tensor(self.noise()).float().to(self.device)
        
        action = np.clip(   action.cpu().detach().numpy(), 
                            self._hp.env.action_min,
                            self._hp.env.action_max)
        self.actor.train()
        return action
        
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
    def _learn_with_per(self):
        if not(self.enough_experience()):
            return
        idxs, experiences, normalized_weights =\
                self.memory.sample()

        states, actions, rewards, new_states, dones = experiences_to_tensor(experiences, self.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        target_actions = self.target_actor(new_states)
        target_values = self.target_critic(new_states, target_actions).flatten()

        y = rewards + self._gamma * target_values * (torch.ones(self._batch_size).to(self.device) - dones)
        values = self.critic(states, actions).flatten()

        td_errors = y - values
        
        self.critic.train()
        self.critic.optimizer.zero_grad()

        normalized_weights = torch.tensor(normalized_weights).float().flatten().to(self.device)
        weighted_td_errors = torch.mul(td_errors, normalized_weights)
        zero_tensor = torch.zeros(weighted_td_errors.shape)
        critic_loss = F.mse_loss(weighted_td_errors, zero_tensor)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()

        mu = self.actor(states)
        actor_loss = -self.critic(states, mu)
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_target_networks()

        priorities = td_errors.abs().flatten().cpu().detach().numpy()
        self.memory.update_priorities(idxs, priorities + 1e-6)

    def _learn_with_uniform_memory(self):
        if not(self.enough_experience()):
            return

        experience_batch = self.memory.sample()
        
        states, actions, rewards, new_states, dones = \
                experiences_to_tensor(experience_batch,self.device)
        
        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        target_actions = self.target_actor(new_states)
        target_values = self.target_critic(new_states, target_actions).flatten()

        y = rewards + self._gamma * target_values * (torch.ones(self._batch_size).to(self.device) - dones)
        values = self.critic(states, actions).flatten()

        self.critic.train()
        self.critic.optimizer.zero_grad()

        critic_loss = F.mse_loss(y, values)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()

        mu = self.actor(states)
        actor_loss = -self.critic(states, mu)
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_target_networks()

    def update_save_path(self, new_path):
        self._save_path = new_path
        self.actor._model_io._save_path = new_path
        self.critic._model_io._save_path = new_path
        self.target_actor._model_io._save_path = new_path
        self.target_critic._model_io._save_path = new_path

    def save_models(self):
        self.actor.save_model()
        self.critic.save_model()
        self.target_actor.save_model()
        self.target_critic.save_model()
        if self._hp.memory.save_checkpoints:
            memory_path = os.path.join(
                self._save_path,
                'memory.pickle'
                )
            with open(memory_path, 'wb') as f:
                memory = self.memory.memory
                pickle.dump(memory, f)
    
    def load_models(self):
        self.actor.load_model()
        self.critic.load_model()
        self.target_actor.load_model()
        self.target_critic.load_model()
