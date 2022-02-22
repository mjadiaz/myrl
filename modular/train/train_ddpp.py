from omegaconf import OmegaConf
from modular.agents.ddpg import DDPG, HyperParamsDDPG
from modular.memory.base import Experience

import gym

hyper_params = HyperParamsDDPG('ddpg.yaml').get_config()

agent = DDPG(hyper_params)

env = gym.make(hyper_params.env.name) 

MAX_EPISODES = 100

for episode in range(MAX_EPISODES):
    agent.noise.reset()
    state = env.reset()
    done = False

    while not(done):
        
        action = agent.select_action(state)

        new_state, reward, done, info = env.step(action)
        
        agent.remember(state, action, reward, new_state, done)
        
        agent.learn()
        state = new_state
    print(agent.memory.__len__())
