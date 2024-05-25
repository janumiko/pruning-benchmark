import logging
from pathlib import Path
import uuid

from architecture.pruning_loop import start_pruning_experiment
from config.main_config import MainConfig
import hydra
from omegaconf import OmegaConf
import torch.multiprocessing as mp

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main_config", version_base="1.2")
def main(cfg: MainConfig) -> None:
    """Main function for the pruning entry point

    Args:
        cfg (MainConfig): Hydra config object with all the settings. (Located in config/main_config.py)
    """
    logger.setLevel(cfg._logging_level)

    logger.info(OmegaConf.to_yaml(cfg))
    hydra_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger.info(f"Hydra output directory: {hydra_output_dir}")

    gpus = cfg._gpus
    if cfg._shared_filesystem:
        ddp_init_method = f"file://{cfg._shared_filesystem}/ddp_init_{uuid.uuid4().hex}"
    else:
        ddp_init_method = "tcp://localhost:12345"
    mp.spawn(
        start_pruning_experiment,
        args=(gpus, cfg, hydra_output_dir, ddp_init_method),
        nprocs=gpus,
        join=True,
    )


if __name__ == "__main__":
    main()
