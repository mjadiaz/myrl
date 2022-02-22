from omegaconf import OmegaConf
from modular.agents.ddpg import DDPG, HyperParamsDDPG
from modular.memory.base import Experience
from modular.common.utils import running_mean
from tqdm import tqdm
import gym
import numpy as np
import os


import matplotlib.pyplot as plt

hyper_params = HyperParamsDDPG('agent_configs/ddpg.yaml').get_config()

agent = DDPG(hyper_params)

env = gym.make(hyper_params.env.name) 

MAX_EPISODES = 1000

total_reward = np.zeros(MAX_EPISODES)

for episode in tqdm(range(1,MAX_EPISODES+1)):
    agent.noise.reset()
    state = env.reset()
    done = False
    
    score = 0

    while not(done):
        
        action = agent.select_action(state)

        new_state, reward, done, info = env.step(action)
        
        agent.remember(state, action, reward, new_state, done)
        
        agent.learn()
        
        state = new_state

        score += reward
        
    total_reward[episode - 1] = score 
    if (episode % 2 == 0):
        

        agent.save_models()
        tmp_total_reward = total_reward[:episode]
        avg_reward = running_mean(tmp_total_reward)
        fig, ax = plt.subplots()
        ax.plot(range(len(tmp_total_reward)), tmp_total_reward)
        ax.plot(range(len(avg_reward)), avg_reward)
        plt.savefig(os.path.join(hyper_params.agent.save_path,'status.png'), dpi=300, bbox_inches='tight')

