from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np

from modular.agents.dqn import DQN, HyperParams
import torch 
import gym

hyper_params = HyperParams('dqn.yaml').get_config()
agent = DQN(hyper_params)
agent.load_models()

env = gym.make(hyper_params.env.name)

MAX_EPISODES = 10

total_reward = []

for episode in tqdm(range(MAX_EPISODES)):
    state = env.reset()
    score = 0
    done = False
    while not(done):
        agent.dqn.eval()
        action = agent.dqn(torch.from_numpy(state).float())
        mas_q_value, action = torch.max(action, dim=0)
        action = action.detach().cpu().numpy()
        new_state, reward, done, info = env.step(action)
        env.render()
        state = new_state
        score += reward

    total_reward.append(score)

total_reward = np.array(total_reward)

        

    
