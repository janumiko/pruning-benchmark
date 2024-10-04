class BasePruningStepScheduler:
    def __init__(self, start: float, steps: int) -> None:
        self.start = start
        self.steps = steps

    def __call__(self, pruning_ratio: float, steps: int) -> list[float]:
        raise NotImplementedError


class OneShotStepScheduler(BasePruningStepScheduler):
    def __call__(self, pruning_ratio: float, steps: int) -> list[float]:
        return [pruning_ratio]
