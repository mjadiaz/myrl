import time
import ray
import numpy as np
import torch
import torch.nn.functional as F

from modular.networks.fc_nets import DDPGActor, DDPGCritic

@ray.remote
class Learner:
    def __init__(self,
            config,
            replay_buffer, 
            parameter_server
            ):
        self.hp = config
        self.replay_buffer = replay_buffer
        self.parameter_server = parameter_server
        self.actor =  DDPGActor(self.hp.actor)
        self.target_actor =  DDPGActor(self.hp.target_actor)
        self.critic = DDPGCritic(self.hp.critic)
        self.target_critic = DDPGCritic(self.hp.target_critic)

        
        self.alpha = self.hp.agent.alpha
        self.beta = self.hp.agent.beta
        self.tau = self.hp.agent.tau
        self.gamma = self.hp.agent.gamma
        self.batch_size = self.hp.agent.batch_size
        self.learning_starts = self.hp.agent.learning_starts
        self.state_dimension = self.hp.env.state_dimension
        self.action_dimension = self.hp.env.action_dimension

        self.total_collected_samples = 0
        self.samples_since_last_update = 0
        
        self.send_weights_to_parameter_server()
        
        self.stopped = False
        
        self.update_target_net()
        self.device = self.actor.device


    def update_target_net(self, tau: float = None):
        '''
        Update the target network parameters. 
        The target network can also slowly track the original network 
        by modifiying the tau parameter.

        Args:
        -----
        tau: float
        '''
        if tau == None:
            tau = self.tau


        def parameters_update(network, target_network, tau=tau):
            net_params = dict(network.named_parameters())
            target_net_params = dict(target_network.named_parameters())

            for name in net_params:  
                net_params[name] = tau*net_params[name].clone() \
                        + (1. - tau)*target_net_params[name].clone()

            target_network.load_state_dict(net_params)

        parameters_update(self.actor, self.target_actor)
        parameters_update(self.critic, self.target_critic)

        


    def send_weights_to_parameter_server(self):
        self.parameter_server.update_weights.remote(self.actor.get_weights())

    def start_learning(self):
        print("Learning starting ... ")
        self.send_weights()
        while not self.stopped:
            sid = self.replay_buffer.get_total_env_samples.remote()
            total_samples = ray.get(sid)

            if total_samples >= self.learning_starts:
                self.optimize()

    def optimize(self):
        samples = ray.get(
                self.replay_buffer.sample.remote())
        if samples:
            N = len(samples)
            self.total_collected_samples += N
            self.samples_since_last_update += N

            obs = np.array(
                    [sample[0] for sample in samples]
                    ).reshape((N, self.state_dimension))
            actions = np.array(
                    [sample[1] for sample in samples]
                    ).reshape((N,self.action_dimension))
            rewards = np.array(
                    [sample[2] for sample in samples]
                    ).reshape((N,))
            last_obs = np.array(
                    [sample[3] for sample in samples]
                    ).reshape((N, self.state_dimension))
            done_flags = np.array(
                    [sample[4] for sample in samples]
                    ).reshape((N,))
            gammas = np.array(
                    [sample[5] for sample in samples]
                    ).reshape((N,))

            obs = torch.tensor(obs).float().to(self.device) 
            actions = torch.tensor(actions).float().to(self.device)    
            rewards = torch.tensor(rewards).float().to(self.device)    
            last_obs = torch.tensor(last_obs).float().to(self.device)    
            dones = torch.tensor(done_flags).float().to(self.device) 
            gammas = torch.tensor(gammas).float().to(self.device) 
            
            # DDPG
            self.target_actor.eval()
            self.target_critic.eval()
            self.critic.eval()

            target_actions = self.target_actor(last_obs)
            target_values = self.target_critic(last_obs, target_actions).flatten()

            y = rewards + gammas * target_values*\
                    (torch.ones(self.batch_size).to(self.device) - dones)
            print(obs.shape, obs)
            print(actions.shape, actions)
            values = self.critic(obs, actions.reshape((actions.shape[0],1))).flatten()

            self.critc.train()
            self.critic.optimizer.zero_grad()

            critic_loss = F.mse_loss(y, values)
            critic_loss.backward()
            self.critic.optimizer.step()

            self.critic.eval()
            self.actor.optimizer.zero_grad()

            mu = self.actor(obs)
            actor_loss = - self.critic(obs, mu)
            actor_loss = torch.mean(actor_loss)
            actor_loss.backward()
            self.actor.optimizer.step()

            self.update_target_net()




            # Double DQN
            #self.target_dqn.eval()
            #self.dqn.eval()
            #q_values = self.dqn(obs)
            #max_q_values = torch.gather(
            #        q_values, 1, actions.unsqueeze(-1).to(torch.int64)) 
            #max_q_values = max_q_values.flatten()
            #astar = torch.argmax(q_values, dim=1)
            #qs = self.target_dqn(last_obs).gather(
            #        dim=1, index=astar.unsqueeze(dim=1)).squeeze()
            #
            #y = rewards + gammas * qs.detach() *\
            #        (torch.ones(self.batch_size).to(self.device) - dones)

            #self.dqn.train()
            #self.dqn.optimizer.zero_grad()
    
            #dqn_loss = F.mse_loss(y, max_q_values)
            #dqn_loss.backward()
            #self.dqn.optimizer.step()


            self.send_weights() 
            
            #if self.samples_since_last_update > 500:
            #     self.update_target_net()
            #     self.samples_since_last_update = 0
            # return True
        else:
            print("No samples received from the buffer.")
            time.sleep(5)
            return False

    def send_weights(self):
        id = self.parameter_server.update_weights.remote(
                self.actor.get_weights()
                )
        ray.get(id)
    def stop(self):
        self.stopped = True
