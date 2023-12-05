class EarlyStopper:
    """Early stopping class.

    Monitors validation loss and stops training if it does not improve after a given patience.
    Improvement is defined as validation loss < min_validation_loss + min_delta.
    Patience is the number of epochs to wait for improvement before stopping.
    """

    def __init__(self, patience: int, min_delta: int) -> None:
        """Initialize EarlyStopper.

        Args:
            patience (int): Number of epochs to wait for improvement.
            min_delta (int): Delta which which is used to decide epoch as improvement.
        """

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss: float) -> bool:
        """Check if training should be stopped.

        Args:
            validation_loss (float): Validation loss.

        Returns:
            bool: Boolean indicating whether training should be stopped.
        """

        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1

        return self.counter >= self.patience
