import torch
from torchmetrics import Metric


class Loss(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("loss", default=torch.tensor(0.0, device=self._device), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0, device=self._device), dist_reduce_fx="sum")

    def update(self, loss, num_samples):
        self.loss += loss
        self.num_samples += num_samples

    def compute(self):
        return self.loss.float() / self.num_samples

    def reset(self):
        self.loss = torch.tensor(0.0, device=self._device)
        self.num_samples = torch.tensor(0, device=self._device)
