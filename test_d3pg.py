import ray
import gym
import numpy as np
import os
from modular.distributed.actor import HyperParamsActor, Actor, ActorEval
from modular.distributed.memory import GlobalMemory
from modular.distributed.parameter_server import ParameterServer
from modular.distributed.learner import Learner
from modular.distributed.utils import Writer

from omegaconf import OmegaConf
#from pheno_game.envs.pheno_env import PhenoEnvContinuous_v0
from modular.memory.memory import DequeReplay, Experience
from ray.tune.registry import register_env
from torch.utils.tensorboard import SummaryWriter

def main():
    #env_config = OmegaConf.load('hep_tools.yaml')
    env_config = None
    hyper_params = HyperParamsActor('d3pg.yaml', env_config=env_config).get_config()
    ray.init(local_mode=False)
    # Writer init
    writer = Writer.remote(hyper_params.agent.save_path)

    #GlobalMemory = ray.remote(DequeReplay)
    global_memory = GlobalMemory.remote(hyper_params.memory)
    # Initialize Parameter Server
    parameter_server = ParameterServer.remote(hyper_params)
    print(ray.get(global_memory.get_memory.remote()))
    #print(ray.get(parameter_server.get_updates_counter.remote()))
<<<<<<< HEAD
    actors  = [Actor.remote(i, parameter_server, global_memory, hyper_params, writer) for i in range(4)]
=======
    actors  = [Actor.remote(i, parameter_server, global_memory, hyper_params, writer) for i in range(15)]
>>>>>>> ee371f64cdd890f9c93c3b15e98bfae74ca01248
    learner = Learner.remote(parameter_server, global_memory, hyper_params, writer)
    # Run all the processes
    processes = []
    for actor in actors:
        processes.append(actor)
    processes.append(learner)
 

    # wait until all processes are done
    processes = [proc.run.remote() for proc in processes]
    ray.wait(processes)

    # print summary
    ray.timeline()

def test(render=False):
    #env_config = OmegaConf.load('hep_tools.yaml')
    env_config = None
    hyper_params = HyperParamsActor('d3pg.yaml', env_config=env_config).get_config()
    ray.init(local_mode=True)

    actor  = ActorEval.remote(hyper_params, n_episodes=10, render=render)
    actor.run.remote()



if __name__ == "__main__":
    test(True)
