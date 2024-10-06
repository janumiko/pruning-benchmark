from architecture.utils.pylogger import RankedLogger
from architecture.utils.training_utils import EarlyStopper, RestoreCheckpoint
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from config.datasets import BaseDataset

logger = RankedLogger(__name__, rank_zero_only=True)


class BaseTrainer:
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
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.dataset_config = dataset_config
        self.epochs = epochs
        self.epochs_per_validation = epochs_per_validation
        self.early_stopper = early_stopper
        self.restore_checkpoint = restore_checkpoint
        self.metrics_logger = metrics_logger
        self.device = device
        self.distributed = distributed

        self._model = None
        self._optimizer = None
        self.current_epoch = 0
        self.total_epochs = 0

        if self.early_stopper.enabled and self.early_stopper.override_epochs_to_inf:
            self.epochs = 10**6

    def _init_ddp(self, model):
        if self.distributed:
            logger.debug("Using DistributedDataParallel")
            if not isinstance(model, DistributedDataParallel):
                model = DistributedDataParallel(model, device_ids=[self.device])
            # TODO: add warning if sampler is not DistributedSampler

        return model

    def fit(self, model, optimizer):
        model = self._init_ddp(model)
        self._model = model
        self._optimizer = optimizer

        logger.info("Initial fit validation")
        resutlts = self.validation_loop()
        self.restore_checkpoint.reset(init_results=resutlts)
        self.early_stopper.reset(init_results=resutlts, next_patience=True)

        logger.info("Starting training loop")
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            self.train_loop()
            if (epoch + 1) % self.epochs_per_validation == 0:
                logger.info(f"Validation after epoch {epoch}")
                resutlts = self.validation_loop()

                self.restore_checkpoint.update(self._model, resutlts)

                if self.early_stopper.check_stop(resutlts):
                    logger.info("Early stopping")
                    break

        self.total_epochs += self.current_epoch + 1
        self.restore_checkpoint.restore_best(self._model)

    def validate(self, model):
        model = self._init_ddp(model)
        self._model = model

        logger.info("Starting validation loop")
        self.validation_loop()

    def train_loop(self):
        raise NotImplementedError

    def validation_loop(self) -> dict[str, float]:
        raise NotImplementedError
