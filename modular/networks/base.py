from abc import ABC, abstractmethod

class NeuralNetBase(ABC):
    
    @property
    @abstractmethod
    def hyper_params(self):
        pass
    
    @property
    @abstractmethod
    def device(self):
        pass


    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def save_model(self):
        pass
    
    @abstractmethod
    def load_model(self):
        pass

