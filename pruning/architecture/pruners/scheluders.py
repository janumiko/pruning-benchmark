class BasePruningScheduler:
    def __init__(self, steps: int):
        self.steps = steps

    def __call__(self, target_sparsity: float, steps: int) -> list[float]:
        """Constructs a schedule for pruning.

        Args:
            target_sparsity (float): Target sparsity level.
            steps (int): Number of pruning steps.

        Returns:
            list[float]: List with pruning values per step.
        """
        raise NotImplementedError


class OneShotPruningScheduler(BasePruningScheduler):
    def __init__(self, steps: int = 1):
        super().__init__(steps)

        if steps > 1:
            raise ValueError("One-shot pruning can only have one step.")

    def __call__(self, target_sparsity: float, steps: int = 1) -> list[float]:
        """Constructs a one-shot schedule for pruning.

        Args:
            target_sparsity (float): Target sparsity level.

        Returns:
            list[float]: List with pruning values per step.
        """

        return [target_sparsity]
