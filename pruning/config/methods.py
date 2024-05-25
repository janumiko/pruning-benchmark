from dataclasses import dataclass


@dataclass
class BasePruningMethodConfig:
    name: str


@dataclass
class LnStructuredConfig(BasePruningMethodConfig):
    name: str = "ln_structured"
    norm: int | str = 1
    dim: int = 0


@dataclass
class GlobalL1UnstructuredConfig(BasePruningMethodConfig):
    name: str = "global_l1_unstructured"
