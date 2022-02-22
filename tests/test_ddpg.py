from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np

from modular.agents.ddpg import DDPG, HyperParamsDDPG
import torch 
import gym

hyper_params = HyperParamsDDPG('ddpg.yaml').get_config()
agent = DDPG(hyper_params)
agent.load_models()

env = gym.make(hyper_params.env.name)

MAX_EPISODES = 10

total_reward = []

for episode in tqdm(range(MAX_EPISODES)):
    state = env.reset()
    score = 0
    done = False
    while not(done):
        agent.actor.eval()
        action = agent.actor(torch.from_numpy(state).float())
        action = action.detach().numpy()
        new_state, reward, done, info = env.step(action)
        env.render()
        state = new_state
        score += reward

    total_reward.append(score)

total_reward = np.array(total_reward)

        

    
