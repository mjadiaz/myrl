from omegaconf import OmegaConf
from modular.agents.ddpg import DDPG, HyperParamsDDPG
from modular.memory.base import Experience
from modular.common.utils import running_mean
from tqdm import tqdm
import gym
import numpy as np
import os
import pandas as pd

import matplotlib.pyplot as plt

from pheno_game.envs.pheno_env import PhenoEnvContinuous_v0
import pickle

env_config  = OmegaConf.load('hep_tools.yaml')
hyper_params = HyperParamsDDPG('ddpg.yaml', env_config=env_config).get_config()

agent = DDPG(hyper_params)
env = gym.make(hyper_params.env.name, env_config=env_config)

MAX_EPISODES = 200
total_reward = np.zeros(MAX_EPISODES)
episode_length = np.zeros(MAX_EPISODES)

progress = pd.DataFrame({
    "total_episode_reward": total_reward,
    "episode_length": episode_length, 
    })

for episode in tqdm(range(1,MAX_EPISODES+1)):
    agent.noise.reset()
    state = env.reset()
    #print("Initial state: ", state)
    done = False
    score = 0
    ep_length = 0

    while not(done):
        action = agent.select_action(state)
        #print(env.denormalize_action(action))
        #print("Agent select action: ", action)
        new_state, reward, done, info = env.step(action)
        #print('New state:' , new_state)
        #print('Reward: ', reward)
        #print("done: ", done)

        agent.remember(state, action, reward, new_state, done)
        agent.learn()
        state = new_state
        score += reward
        ep_length += 1
    progress.total_episode_reward[episode-1] = score
    progress.episode_length[episode-1] = ep_length
    #env.close()

    if (episode % 25 == 0):
        chckpt_path = 'runs/chckpt_nr_'+str(episode)
        agent.update_save_path(chckpt_path)
        agent.save_models()

        #memory_path = os.path.join(
        #    hyper_params.agent.save_path,
        #    'memory.pickle'
        #    )
        progress_path = os.path.join(
            chckpt_path,
            'progress.csv'
            )
        progress[:episode].to_csv(progress_path)
        #with open(memory_path, 'wb') as f:
        #    memory = agent.memory.memory
        #    pickle.dump(memory, f)




