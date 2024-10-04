from architecture.trainers import Trainer
import torch
from torch import nn
from torch.utils.data import DataLoader
from architecture.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class CVTrainer(Trainer):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_fn = nn.CrossEntropyLoss()

    def fit(self, model, optimizer):
        super().fit(model, optimizer)

    def train_loop(self):
        self.model.to(self.device)
        self.model.train()

        for data, target in self.train_dataloader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()

    @torch.no_grad
    def validation_loop(self):
        self.model.to(self.device)
        self.model.eval()

        val_loss = 0
        correct = 0

        for data, target in self.val_dataloader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            val_loss += self.loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(self.val_dataloader.dataset)
        accuracy = 100.0 * correct / len(self.val_dataloader.dataset)
        print(f"Validation loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")

        return {"loss": val_loss, "accuracy": accuracy}
