import logging
from pathlib import Path

from architecture.pruning_loop import start_pruning_experiment
import architecture.utility as utility
from config.main_config import MainConfig
import hydra
from omegaconf import OmegaConf
import wandb

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main_config", version_base="1.2")
def main(cfg: MainConfig) -> None:
    """Main function for the pruning entry point

    Args:
        cfg (MainConfig): Hydra config object with all the settings. (Located in config/main_config.py)
    """
    logger.info(OmegaConf.to_yaml(cfg))
    hydra_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger.info(f"Hydra output directory: {hydra_output_dir}")

    if cfg._seed.is_set:
        utility.training.set_reproducibility(cfg._seed.value)

    wandb_run = wandb.init(
        project=cfg._wandb.project if cfg._wandb.logging else None,
        mode="disabled" if not cfg._wandb.logging else "online",
    )

    start_pruning_experiment(cfg, hydra_output_dir, wandb_run)


if __name__ == "__main__":
    main()
