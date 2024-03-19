import pandas as pd
import logging
from pathlib import Path
from omegaconf import OmegaConf
from config.main_config import MainConfig

logger = logging.getLogger(__name__)


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


def save_results_with_config(
    results_df: pd.DataFrame,
    cfg: MainConfig,
    output_path: Path,
) -> None:
    """Save the results dataframe with the configuration settings to a csv file.
    It skips keys starting with underscore from the Hydra configuration.

    Args:
        results_df (pd.DataFrame): Dataframe with the results.
        cfg (MainConfig): Hydra configuration object (dataclass based).
        output_dir (Path): Path to the output directory.
        current_date_str (str): A string with the current date and time.
    """

    dict_config = OmegaConf.to_container(cfg, resolve=True)
    dict_config = strip_underscore_keys(dict_config)
    dict_config = pd.json_normalize(dict_config)
    logger.info(f"Normalized dictionary config {dict_config}")

    results_df = pd.concat([dict_config, results_df], axis=1)
    results_df = results_df.replace("???", None)

    logger.info(f"Saving results to {output_path}")
    results_df.to_csv(
        output_path,
        mode="w",
        header=True,
        index=True,
    )
