import ray
from modular.networks.fc_nets import DDPGActor, DDPGCritic

@ray.remote
class ParameterServer:
    def __init__(self, actor, critic):
        self.weights = None
        self.eval_weights = None
        self.actor =  actor

        self.critic_weights = None
        self.critic = critic

    def update_weights(self, new_parameters):
        self.weights = new_parameters
        return True
        
    def update_critic_weights(self, new_parameters):
        self.critic_weights = new_parameters
        return True

    def get_weights(self):
        return self.weights

    def get_critic_weights(self):
        return self.critic_weights

    def get_eval_weights(self):
        return self.eval_weights

    def set_eval_weights(self):
        self.eval_weights = self.weights
        return True

    def save_eval_weights(self,
                          filename=
                          'checkpoints/model_checkpoint'):
        self.actor.set_weights(self.eval_weights)
        self.actor.save_model()

        self.critic.set_weights(self.critic_weights)
        self.critic.save_model()
        print("Saved.")

