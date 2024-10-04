import torch.distributed as dist


def init_process_group(
    rank: int,
    world_size: int,
    init_method: str,
    backend: str = "nccl",
):
    """Initialize the distributed process group."""
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    dist.barrier()


def get_rank() -> int:
    """Get the rank of the current process."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def distributed_only(fn):
    """Decorate a function to only run on distributed processes."""

    def wrapped_fn(*args, **kwargs):
        if dist.is_available() and dist.is_initialized():
            return fn(*args, **kwargs)
        return None

    return wrapped_fn


def rank_zero_only(fn):
    """Decorate a function to only run on the rank zero process."""

    def wrapped_fn(*args, **kwargs):
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            if rank == 0:
                return fn(*args, **kwargs)
        return fn(*args, **kwargs)

    return wrapped_fn
