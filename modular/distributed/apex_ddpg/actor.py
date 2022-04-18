from collections import deque
import ray
import gym 
import numpy as np
import torch


from modular.networks.fc_nets import DDPGActor
from modular.exploration.noises import OUActionNoise, GaussNoise
@ray.remote
class Actor:
    def __init__(self,
            actor_id,
            replay_buffer,
            parameter_server,
            config,
            eps,
            eval=False):
        
        self.actor_id = actor_id
        self.replay_buffer = replay_buffer
        self.parameter_server = parameter_server
        self.hp = config
        self.gamma = self.hp.agent.gamma
        self.eps = eps
        self.eval = eval
        self.actor= DDPGActor(self.hp.actor)

        self.device = self.actor.device
        self.env = gym.make(self.hp.env.name)
        self.local_buffer = []
        self.state_dimension = self.hp.env.state_dimension
        self.action_dimension = self.hp.env.action_dimension
        self.multi_step_n = self.hp.agent.multi_step_n # = 1
        self.q_update_freq = self.hp.agent.q_update_freq # = 100
        self.send_experience_freq = \
                self.hp.agent.send_experience_freq # 100
        self.continue_sampling = True
        self.current_episodes = 0
        self.current_steps = 0
        self.noise_type = self.hp.agent.noise_type
        if self.noise_type == 'OU':
            self.noise = OUActionNoise(mu=np.zeros(self.action_dimension))
        else:
            self.noise = GaussNoise(scale=1, size=self.action_dimension)
        


    def update_actor_network(self):
        if self.eval:
            pid =\
                self.parameter_server.get_eval_weights.remote()
        else:
            pid =\
                self.parameter_server.get_weights.remote()
        new_weights = ray.get(pid)
        if new_weights:
            self.actor.set_weights(new_weights)
        else:
            print("Weights are not available yet, skipping.")

    @torch.no_grad()
    def get_action(self, state: np.ndarray):
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
        self.actor.eval()
        state = torch.tensor(state).float().to(self.device)
        action = self.actor(state) +\
                torch.tensor(self.noise()).float().to(self.device)
        action = np.clip(
                action.cpu().detach().numpy(),
                self.hp.env.action_min,
                self.hp.env.action_max
                )
        self.actor.train()
        return action

    def get_n_step_trans(self, n_step_buffer):
        discounted_return = 0
        power_gamma = 1
        for transition in list(n_step_buffer)[:-1]:
            _, _, reward, _ = transition
            discounted_return += power_gamma * reward
            power_gamma *= self.gamma
        observation, action, _, _ = n_step_buffer[0]
        last_observation, _, _, done = n_step_buffer[-1]
        experience = (observation, action, discounted_return,
                last_observation, done, power_gamma)
        return experience
    def stop(self):
        self.continue_sampling = False
    
    def sample(self):
        print("Starting sampling in actor {}".format(self.actor_id))
        self.update_actor_network()
        observation = self.env.reset()
        episode_reward = 0 
        episode_length = 0
        n_step_buffer = deque(maxlen=self.multi_step_n + 1)
        while self.continue_sampling:
            action = self.get_action(observation)
            next_observation, reward,\
                    done, info = self.env.step(action)
            n_step_buffer.append((observation, action, reward, done))

            if len(n_step_buffer) == self.multi_step_n + 1:
                self.local_buffer.append(
                        self.get_n_step_trans(n_step_buffer)
                        )
            self.current_steps += 1
            episode_reward += reward
            episode_length += 1
            if done:
                if self.eval:
                    break
                next_observation = self.env.reset()
                if len(n_step_buffer) > 1:
                    self.local_buffer.append(
                            self.get_n_step_trans(n_step_buffer)
                            )
                self.current_episodes += 1
                episode_reward = 0
                episode_length = 0
            observation = next_observation
            if self.current_steps %\
                    self.send_experience_freq == 0 and not self.eval:
                        self.send_experience_to_replay()
            if self.current_steps % \
                    self.q_update_freq == 0 and not self.eval:
                        self.update_actor_network()
        return episode_reward

    def send_experience_to_replay(self):
        rf = self.replay_buffer.add.remote(self.local_buffer)
        ray.wait([rf])
        self.local_buffer = []



