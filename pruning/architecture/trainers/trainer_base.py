import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        epochs: int,
        epochs_per_validation: int,
        device: torch.device,
        metrics_logger=None,
        ddp_strategy: bool = False,
    ):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.epochs = epochs
        self.epochs_per_validation = epochs_per_validation
        self.metrics_logger = metrics_logger
        self.device = device
        self.ddp_strategy = ddp_strategy

        self.model = None
        self.optimizer = None

    def _init_ddp(self, model):
        if self.ddp_strategy:
            if not isinstance(model, DistributedDataParallel):
                model = DistributedDataParallel(model, device_ids=[self.device])
            # TODO: add warning if sampler is not DistributedSampler

        return model

    def fit(self, model, optimizer):
        model = self._init_ddp(model)
        self.model = model
        self.optimizer = optimizer

        for epoch in range(self.epochs):
            self._train_loop()
            if (epoch + 1) % self.epochs_per_validation == 0:
                self.validate(self.model)

    def validate(self, model):
        model = self._init_ddp(model)
        self.model = model

        self._validation_loop()

    def train_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def on_train_start(self):
        pass

    def on_train_end(self):
        pass

    def on_validation_start(self):
        pass

    def on_validation_end(self):
        pass

    def _train_loop(self):
        self.on_train_start()
        for batch_idx, batch in enumerate(self.train_dataloader):
            self.train_step(batch, batch_idx)
        self.on_train_end()

    def _validation_loop(self):
        self.on_validation_start()
        for batch_idx, batch in enumerate(self.val_dataloader):
            self.validation_step(batch, batch_idx)
        self.on_validation_end()
