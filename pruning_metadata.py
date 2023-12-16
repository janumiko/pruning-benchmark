import torch.nn.utils.prune as prune
from dataclasses import dataclass


@dataclass
class PruningMetadata:
    total_pruned: float
    pruning_step: float
    finetune_epochs: int
    method: prune.BasePruningMethod
    early_stopping: bool
