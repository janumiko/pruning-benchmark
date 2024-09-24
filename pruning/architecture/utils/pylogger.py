import logging
from typing import Mapping

import torch.distributed as dist

from architecture.utils.ddp import get_rank


def rank_prefixed_message(message: str, rank: int | None) -> str:
    """Add a prefix with the rank to a message."""
    if rank is not None:
        return f"[rank: {rank}] {message}"
    return message


class RankedLogger(logging.LoggerAdapter):
    """A multi-GPU-friendly python command line logger."""

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        """Initializes a multi-GPU-friendly python command line logger that logs on all processes
        with their rank prefixed in the log message.

        Args:
            name: The name of the logger.
            rank_zero_only: If `True`, then the log will only occur on the rank zero process.
            extra: Extra attributes to add to the log record.
        """
        logger = logging.getLogger(name)
        super().__init__(logger=logger, extra=extra)
        self.rank_zero_only = rank_zero_only

    def log(self, level: int, msg: str, rank: int | None = None, *args, **kwargs) -> None:
        """Delegate a log call to the underlying logger, after prefixing its message with the rank
        of the process it's being logged from. If `'rank'` is provided, then the log will only
        occur on that rank/process.

        Args:
            level: The log level.
            msg: The log message.
            rank: The rank of the process to log from. If `None`, then the log will occur on all processes.
            *args: Additional arguments to pass to the logger.
            **kwargs: Additional keyword arguments to pass to the logger.
        """
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)

            if dist.is_available() and dist.is_initialized():
                current_rank = get_rank()
            else:
                current_rank = None

            msg = rank_prefixed_message(msg, current_rank)
            if current_rank is None:
                self.logger.log(level, msg, *args, **kwargs)
            elif self.rank_zero_only:
                if current_rank == 0:
                    self.logger.log(level, msg, *args, **kwargs)
            elif rank is None or current_rank == rank:
                self.logger.log(level, msg, *args, **kwargs)
