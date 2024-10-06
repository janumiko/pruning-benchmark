from architecture.trainers import BaseTrainer
from architecture.utils.metrics import Loss
from architecture.utils.pylogger import RankedLogger
from architecture.utils.training_utils import EarlyStopper, RestoreCheckpoint
from config.datasets import BaseDataset
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

logger = RankedLogger(__name__, rank_zero_only=True)


class ClassificationTrainer(BaseTrainer):
    def __init__(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        dataset_config: BaseDataset,
        epochs: int,
        epochs_per_validation: int,
        early_stopper: EarlyStopper,
        restore_checkpoint: RestoreCheckpoint,
        device: torch.device,
        metrics_logger=None,
        distributed: bool = False,
    ):
        super().__init__(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            dataset_config=dataset_config,
            epochs=epochs,
            epochs_per_validation=epochs_per_validation,
            early_stopper=early_stopper,
            restore_checkpoint=restore_checkpoint,
            device=device,
            metrics_logger=metrics_logger,
            distributed=distributed,
        )
        self.loss_fn = nn.CrossEntropyLoss()

        # Metrics
        self.eval_loss = Loss().to(self.device)
        self.eval_accuracy = Accuracy(task="multiclass", num_classes=self.dataset_config.num_classes).to(self.device)

    def fit(self, model, optimizer):
        super().fit(model, optimizer)

    def train_loop(self):
        self._model.to(self.device)
        self._model.train()

        for data, target in self.train_dataloader:
            data, target = data.to(self.device), target.to(self.device)
            self._optimizer.zero_grad()
            output = self._model(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            self._optimizer.step()

    @torch.no_grad
    def validation_loop(self):
        self._model.to(self.device)
        self._model.eval()

        self.eval_accuracy.reset()
        self.eval_loss.reset()

        for data, target in self.val_dataloader:
            data, target = data.to(self.device), target.to(self.device)
            output = self._model(data)
            val_loss = self.loss_fn(output, target)
            self.eval_loss.update(val_loss, data.size(0))
            self.eval_accuracy.update(output, target)

        val_loss = self.eval_loss.compute()
        accuracy = self.eval_accuracy.compute()
        print(f"Validation loss: {val_loss:.4f}, Accuracy: {(accuracy*100):.2f}%")

        return {"loss": val_loss, "accuracy": accuracy}
