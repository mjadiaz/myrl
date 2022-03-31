import ray
import torch
from modular.networks.fc_nets import DDPGActor
from omegaconf import OmegaConf, DictConfig

@ray.remote
class ParameterServer:
    def __init__(self, hyper_params: DictConfig):
        self.hp = hyper_params
        
        # Create copy of the actor network
        self.actor = DDPGActor(self.hp.actor)
        self.actor.device = 'cpu'

        # Updates counter
        self.updates_counter = 0

        # Learning steps counter
        self.learning_steps = 0
        
    def increment_updates_counter(self):
        self.updates_counter += 1
        #print(f'param_server: we have {self.updates_counter} update so far mate')
    def increment_learning_steps_counter(self):
        self.learning_steps += 1
    def get_learning_steps_counter(self):
        return self.learning_steps
    
    def get_updates_counter(self):
        return self.updates_counter

    def get_weights(self):
        #weights = dict(self.actor.named_parameters())
        weights = self.actor.state_dict()
        return weights

    def update_weights(self, weights):
        self.actor.load_state_dict(weights)
        self.increment_updates_counter()
        #print('weights updated')






