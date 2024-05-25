import logging
from pathlib import Path

from architecture.pruning_loop import start_pruning_experiment
from config.main_config import MainConfig
import hydra
from omegaconf import OmegaConf
import torch.multiprocessing as mp
import uuid

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
    uuid_str = uuid.uuid4().hex
    mp.spawn(start_pruning_experiment, args=(gpus, cfg, hydra_output_dir, uuid_str), nprocs=gpus, join=True)


if __name__ == "__main__":
    main()
