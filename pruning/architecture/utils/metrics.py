from typing import Literal, Sequence, Mapping

from architecture.utils import distributed_utils
from architecture.utils.pylogger import RankedLogger
import torch
from torchmetrics import Metric
import wandb
from wandb.sdk.wandb_run import Run
from omegaconf import OmegaConf


logger = RankedLogger(__name__, rank_zero_only=True)


class BaseMetricLogger:
    @distributed_utils.rank_zero_only
    def log(self, data: dict, commit: bool = True):
        raise NotImplementedError

    @distributed_utils.rank_zero_only
    def summary(self, data: dict):
        raise NotImplementedError

    @distributed_utils.rank_zero_only
    def finish(self):
        raise NotImplementedError

    @distributed_utils.rank_zero_only
    def define_metrics(self, metrics: dict[str, Literal["max", "min", "mean", "last"]]):
        raise NotImplementedError


class WandbMetricLogger(BaseMetricLogger):
    def __init__(
        self,
        mode: Literal["disabled", "offline", "online"] = "disabled",
        main_config: Mapping | None = None,
        project: str | None = None,
        entity: str | None = None,
        run_name: str | None = None,
        tags: Sequence[str] | None = None,
        group: str | None = None,
        job_type: str | None = None,
        log_path: str | None = None,
    ):
        self.mode = mode
        self.project = project
        self.entity = entity
        self.run_name = run_name
        self.tags = tags
        self.main_config = OmegaConf.to_container(main_config)
        self.group = group
        self.job_type = job_type
        self.log_path = log_path

        self._run: Run = self._init_wandb()

    @distributed_utils.rank_zero_only
    def _init_wandb(self) -> Run:
        logger.info("Initializing wandb")
        return wandb.init(
            project=self.project,
            entity=self.entity,
            mode=self.mode,
            name=self.run_name,
            tags=self.tags,
            group=self.group,
            job_type=self.job_type,
            dir=self.log_path,
            config=self.main_config,
        )

    @distributed_utils.rank_zero_only
    def log(self, data: dict, commit: bool = True):
        self._run.log(data, commit=commit)

    @distributed_utils.rank_zero_only
    def summary(self, data: dict):
        self._run.summary.update(data)

    @distributed_utils.rank_zero_only
    def finish(self):
        logger.info("Finishing wandb run")
        self._run.finish()

    @distributed_utils.rank_zero_only
    def define_metrics(self, metrics: dict[str, Literal["max", "min", "mean", "last"]]):
        for metric_name, reduction in metrics.items():
            self._run.define_metric(metric_name, summary=reduction)


class Loss(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state(
            "loss", default=torch.tensor(0.0, device=self._device), dist_reduce_fx="sum"
        )
        self.add_state(
            "num_samples", default=torch.tensor(0, device=self._device), dist_reduce_fx="sum"
        )

    def update(self, loss, num_samples):
        self.loss += loss
        self.num_samples += num_samples

    def compute(self):
        return self.loss.float() / self.num_samples

    def reset(self):
        self.loss = torch.tensor(0.0, device=self._device)
        self.num_samples = torch.tensor(0, device=self._device)
