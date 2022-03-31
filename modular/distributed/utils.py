import ray

from torch.utils.tensorboard import SummaryWriter


@ray.remote
class Writer:
    def __init__(self, run_name):
        self.run_name = run_name
        self.writer  = SummaryWriter(self.run_name)

    def add_scalar(self, name, value, n_iter):
        self.writer.add_scalar(name, value, n_iter)
