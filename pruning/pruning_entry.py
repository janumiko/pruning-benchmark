import logging
from pathlib import Path

from architecture.pruning_loop import start_pruning_experiment
import architecture.utility as utility
from config.main_config import MainConfig
import hydra
from omegaconf import OmegaConf

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

    if cfg._seed.is_set:
        utility.training.set_reproducibility(cfg._seed.value)

    start_pruning_experiment(cfg, hydra_output_dir)


if __name__ == "__main__":
    main()
