from torch.utils.tensorboard import SummaryWriter
import ray
from omegaconf import OmegaConf, DictConfig
import gym

class Writer:
    def __init__(self, run_name):
        self.run_name = run_name
        self.writer  = SummaryWriter(self.run_name)

    def add_scalar(self, name, value, n_iter):
        self.writer.add_scalar(name, value, n_iter)


class HyperParams:
    def __init__(self,
            agent_file: str,
            save_path=None,
            env_config: DictConfig = None
            ):
        self._hp = OmegaConf.load(agent_file)
        self.env_config = env_config
        if not save_path == None:
            self._hp.agent.save_path = save_path
        self.alpha = self._hp.agent.alpha
        self.beta = self._hp.agent.beta
        self.gamma = self._hp.agent.gamma
        self.create_env() 
        self.create_network('actor_server', self.alpha )
        self.create_network('actor', self.alpha )
        self.create_network('target_actor', self.alpha )
        self.create_network('critic', self.beta )
        self.create_network('target_critic', self.beta )
        self.create_memory()
        
    def __repr__(self):
        return str(self._hp)
    
    def get_config(self):
        return self._hp
    def create_env(self):
        if not(self.env_config == None):
            env = gym.make(self._hp.name, env_config=self.env_config)
        else:
            env = gym.make(self._hp.env.name)
        config = OmegaConf.create(
                {'env': {
                    'state_dimension': env.observation_space.shape[0],
                    'action_dimension': env.action_space.shape[0],
                    'action_min': env.action_space.low.tolist(),
                    'action_max': env.action_space.high.tolist()
                    }})
        self._hp = OmegaConf.merge(self._hp, config)
    def create_network(self, network: str, lr: float):
        '''
        Create a config for the network and merge 
        with agent config
        '''
        config = OmegaConf.create(
                {network: { 
                    'name': network, 
                    'save_path': self._hp.agent.save_path,
                    'learning_rate': lr,
                    'state_dimension': self._hp.env.state_dimension,
                    'action_dimension': self._hp.env.action_dimension}})
        self._hp = OmegaConf.merge(self._hp, config)
    def create_memory(self):
        config = OmegaConf.create({
            'memory':{
                'batch_size': self._hp.agent.batch_size,
                'max_size': self._hp.agent.max_size,
                'action_dimension': self._hp.env.action_dimension,
                'state_dimension': self._hp.env.state_dimension}})
        self._hp = OmegaConf.merge(self._hp, config)


