import ray

from omegaconf import OmegaConf, DictConfig

from modular.networks.fc_nets import DDPGActor, DDPGCritic
from modular.common.utils import experiences_to_tensor
from modular.memory.memory import Experience

import torch
import torch.nn.functional as F
import pickle
import os
import gym


@ray.remote
class Learner:
    def __init__(
        self,
        parameter_server,
        global_memory,
        hyper_params
        ):
        
        self.hp = hyper_params
        self.parameter_server = parameter_server
        self.global_memory = global_memory
        
        # Learner parameters
        self.alpha = self.hp.agent.alpha
        self.beta = self.hp.agent.beta
        self.tau = self.hp.agent.tau
        self.gamma = self.hp.agent.gamma
        self.state_dimension = self.hp.env.state_dimension
        self.action_dimension = self.hp.env.action_dimension
        self.max_steps = self.hp.agent.max_steps
        self.save_path = self.hp.agent.save_path
        self.batch_size = self.hp.agent.batch_size
        self.update_params_interval = self.hp.agent.update_params_interval

        # Learning steps counter
        self.learning_steps = 0

        # Create networks
        # Do I need actor? I don't think so
        self.actor = DDPGActor(self.hp.actor)
        self.target_actor =  DDPGActor(self.hp.target_actor)
        self.critic = DDPGCritic(self.hp.critic)
        self.target_critic = DDPGCritic(self.hp.target_critic)

        # Copy networks parameters to target nteworks (tau=1)
        self.update_target_networks(tau=1.)

        # Device
        self.device = self.critic.device

    def update_target_networks(self, tau: float = None):
        '''Update the parameters of the target networks'''
        if tau == None:
            tau = self.tau

        def parameters_update(network, target_network, tau=tau):
            net_params = dict(network.named_parameters())
            target_net_params = dict(target_network.named_parameters())

            for name in net_params:
                net_params[name] = tau*net_params[name].clone()\
                    + (1. - tau)*target_net_params[name].clone()
            target_network.load_state_dict(net_params)
        parameters_update(self.actor, self.target_actor)
        parameters_update(self.critic, self.target_critic)
    
    def able_to_learn(self):
        #print("learner: checking if able to learn from memory")
        return ray.get(self.global_memory.able_to_learn.remote())

    def _learn_with_uniform_memory(self):
        if not self.able_to_learn():
            #print('learner: not yet learning')
            return
        #print('learning')
        experience_batch = ray.get(
            self.global_memory.sample.remote()
            )
        states, actions, rewards, new_states, dones =\
            experiences_to_tensor(experience_batch, self.device)
        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        target_actions = self.target_actor(new_states)
        target_values = self.target_critic(new_states, target_actions).flatten()
        #print('target values: ',target_values)
        y = rewards + self.gamma * target_values *\
            (torch.ones(self.batch_size).to(self.device) - dones)
        values = self.critic(states, actions).flatten()
        #print('values: ', values)
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
        self.learning_steps += 1
        #print('learning steps, ',self.learning_steps)
    
    def get_learning_steps(self):
        return self.learning_steps

    def learn(self):
        self._learn_with_uniform_memory()

    def update_save_path(self, new_path):
        self.save_path = new_path
        self.actor._model_io._save_path = new_path
        self.critic._model_io._save_path = new_path
        self.target_actor._model_io._save_path  = new_path
        self.target_critic._model_io._save_path = new_path

    def save_models(self):
        self.actor.save_models()
        self.critic.save_models()
        self.target_actor.save_models()
        self.target_critic.save_models()
        if self.hp.memory.save_checkpoints:
            memory_path = os.path.join(
                self.save_path,
                'memory.pickle'
                )
            with open(memory_path, 'wb') as f:
                memory = ray.get(self.global_memory.get_memory.remote())
                pickle.dump(memory, f)

    def load_models(self):
        self.actor.load_model()
        self.critic.load_model()
        self.target_critic.load_model()
        self.target_actor.load_model()

    def push_actor_parameters(self):
        actor_parameters = self.actor.state_dict()
        self.parameter_server.update_weights.remote(actor_parameters)

    def run(self):
        while ray.get(self.global_memory.get_step_counter.remote()) < self.max_steps:
            #print(f'learner: empece que wea, paso {self.get_learning_steps()}')
            self.learn()
            
            if (self.learning_steps % self.update_params_interval == 0) and (self.learning_steps > 0):
                self.push_actor_parameters()





