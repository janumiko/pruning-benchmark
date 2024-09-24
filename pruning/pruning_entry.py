from pathlib import Path

from architecture.utils.pylogger import RankedLogger
from config.main_config import MainConfig
import hydra
from omegaconf import OmegaConf
import torch.multiprocessing as mp
from architecture.utils.distributed import init_process_group
from architecture.pruning import start_pruning_experiment

logger = RankedLogger(__name__, rank_zero_only=True)


def start_process(rank: int, cfg: MainConfig) -> None:
    init_process_group(
        rank=rank,
        world_size=cfg.distributed.gpu,
        init_method=cfg.distributed.init_method,
    )
    start_pruning_experiment(cfg)


@hydra.main(config_path="config", config_name="main_config", version_base="1.3")
def main(cfg: MainConfig) -> None:
    """Main function for the pruning entry point

    Args:
        cfg (MainConfig): Hydra config object with all the settings. (Located in config/main_config.py)
    """
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info(f"Hydra output directory: {cfg.paths.output_dir}")

    if cfg.distributed.enabled:
        mp.spawn(
            start_process,
            args=(cfg,),
            nprocs=cfg.distributed.gpu,
            join=True,
        )
    else:
        start_pruning_experiment(cfg)


if __name__ == "__main__":
    main()
