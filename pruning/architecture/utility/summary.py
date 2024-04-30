import logging
from pathlib import Path
from typing import Mapping

from config.main_config import MainConfig
from omegaconf import OmegaConf
import pandas as pd
import wandb

logger = logging.getLogger(__name__)


def get_run_group_name(cfg: MainConfig, current_date_str: str) -> str:
    """Create a group name for the runs based on the configuration settings.

    Args:
        cfg (MainConfig): Hydra configuration object (dataclass based).
        current_date_str (str): A string with the current date and time.

    Returns:
        str: A string with the group name for the runs.
    """
    run_name = (
        f"{cfg.model}_"
        f"{cfg.dataset.name}_"
        f"{cfg.pruning.scheduler.name}_"
        f"{cfg.pruning.method.name}_"
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


def strip_underscore_keys(input_dict: Mapping) -> dict:
    """Creates a new dictionary without keys starting with underscore.
    In case the value is a dictionary, it will call itself recursively.

    Args:
        input_dict (Mapping): A input dictionary to filter underscore prefix keys.

    Returns:
        dict: A dictionary without keys starting with underscore.
    """
    filtered_dict = {}

    for key, value in input_dict.items():
        if key.startswith("_"):
            continue

        if isinstance(value, Mapping) and (filtered_value := strip_underscore_keys(value)):
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


def create_wandb_run(cfg: MainConfig, group_name: str, run_name: str) -> wandb.sdk.wandb_run.Run:
    """Create a W&B run based on the configuration settings.
    In case logging is disabled, it will create a dry-run.

    Args:
        cfg (MainConfig): Hydra configuration object (dataclass based).
        group_name (str): Group name for the run to belong.
        run_name (str): Name of the run.

    Returns:
        wandb.sdk.wandb_run.Run: A W&B run object.
    """

    config = strip_underscore_keys(OmegaConf.to_container(cfg, resolve=True))

    if cfg._wandb.logging:
        wandb_run = wandb.init(
            project=cfg._wandb.project,
            mode="online",
            group=group_name,
            name=run_name,
            job_type=cfg._wandb.job_type,
            entity=cfg._wandb.entity,
            config=config,
        )
    else:
        wandb_run = wandb.init(mode="disabled")

    return wandb_run


def save_checkpoint_results(
    cfg: MainConfig,
    results: pd.DataFrame,
    out_directory: Path,
    group_name: str,
    base_top1acc: float = 0.0,
    base_top5acc: float = 0.0,
) -> None:
    """Save the results dataframe to a csv file and log it to W&B.

    Args:
        cfg (MainConfig): Hydra configuration object (dataclass based).
        results (pd.DataFrame): Dataframe with the results.
        out_directory (Path): Path to the output directory.
        group_name (str): Group name for the run to belong.
        base_top1acc (float, optional): Base top-1 accuracy. Defaults to 0.0.
        base_top5acc (float, optional): Base top-5 accuracy. Defaults to 0.0.
    """
    results.to_csv(f"{out_directory}/pruning_results.csv", index=False, float_format="%.4f")

    wandb_run = create_wandb_run(cfg, group_name, "pruning_results")
    summary = wandb_run.summary
    summary["base_top1_accuracy"] = base_top1acc
    summary["base_top5_accuracy"] = base_top5acc

    table = wandb.Table(dataframe=results)
    artifact = wandb.Artifact(f"{group_name}_pruning_results", type="results")
    artifact.add(table, "pruning_results")
    wandb_run.log_artifact(artifact)
    wandb_run.finish()
