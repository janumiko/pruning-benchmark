import logging

from config.main_config import MainConfig
from omegaconf import OmegaConf
import pandas as pd

logger = logging.getLogger(__name__)


def get_run_name(cfg: MainConfig, current_date_str: str) -> str:
    """Create a run name based on the configuration settings

    Args:
        cfg (MainConfig): Hydra configuration object (dataclass based).
        current_date_str (str): A string with the current date and time.

    Returns:
        str: A string with the run name.
    """
    run_name = (
        f"{cfg.model.name}_"
        f"{cfg.dataset.name}_"
        f"{cfg.pruning.iteration_rate}-"
        f"{cfg.pruning.iterations}-"
        f"{cfg.pruning.finetune_epochs}_"
        f"{current_date_str}"
    )
    return run_name


def log_summary(results_df: pd.DataFrame) -> None:
    """Log the summary of the results

    Args:
        results_df (pd.DataFrame): The results dataframe
    """

    acc_mean = results_df["top-1 accuracy"].mean()
    acc_std = results_df["top-1 accuracy"].std()
    acc_diff_mean = results_df["top-1 difference"].mean()
    acc_diff_std = results_df["top-1 difference"].std()

    top5_mean = results_df["top-5 accuracy"].mean()
    top5_std = results_df["top-5 accuracy"].std()
    top5_diff_mean = results_df["top-5 difference"].mean()
    top5_diff_std = results_df["top-5 difference"].std()

    logger.info(f"Mean top-1 accuracy {acc_mean:.2f}% ± {acc_std:.2f}%")
    logger.info(f"Mean top-1 difference {acc_diff_mean:.2f}% ± {acc_diff_std:.2f}%")
    logger.info(f"Mean top-5 accuracy {top5_mean:.2f}% ± {top5_std:.2f}%")
    logger.info(f"Mean top-5 difference {top5_diff_mean:.2f}% ± {top5_diff_std:.2f}%")


def strip_underscore_keys(input_dict: dict) -> dict:
    """Creates a new dictionary without keys starting with underscore.
    In case the value is a dictionary, it will call itself recursively.

    Args:
        input_dict (dict): A input dictionary to filter underscore prefix keys.

    Returns:
        dict: A dictionary without keys starting with underscore.
    """
    filtered_dict = {}

    for key, value in input_dict.items():
        if key.startswith("_"):
            continue

        if isinstance(value, dict) and (filtered_value := strip_underscore_keys(value)):
            filtered_dict[key] = filtered_value
        else:
            filtered_dict[key] = value

    return filtered_dict


def create_config_dataframe(
    cfg: MainConfig,
) -> pd.DataFrame:
    """Save the results dataframe with the configuration settings to a csv file.
    It skips keys starting with underscore from the Hydra configuration.

    Args:
        results_df (pd.DataFrame): Dataframe with the results.
        cfg (MainConfig): Hydra configuration object (dataclass based).
        output_dir (Path): Path to the output directory.
        current_date_str (str): A string with the current date and time.
    """

    dict_config = strip_underscore_keys(OmegaConf.to_container(cfg, resolve=True))
    logger.info(f"Normalized dictionary config {dict_config}")
    config_df = pd.json_normalize(dict_config)
    config_df = config_df.replace("???", None)

    return config_df
