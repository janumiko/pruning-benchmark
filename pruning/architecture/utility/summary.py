import pandas as pd
import logging

logger = logging.getLogger(__name__)


def log_summary(results_df: pd.DataFrame) -> None:
    """Log the summary of the results

    Args:
        results_df (pd.DataFrame): The results dataframe
    """

    acc_mean = results_df["Top-1 accuracy"].mean()
    acc_std = results_df["Top-1 accuracy"].std()
    acc_diff_mean = results_df["Top-1 difference"].mean()
    acc_diff_std = results_df["Top-1 difference"].std()

    top5_mean = results_df["Top-5 accuracy"].mean()
    top5_std = results_df["Top-5 accuracy"].std()
    top5_diff_mean = results_df["Top-5 difference"].mean()
    top5_diff_std = results_df["Top-5 difference"].std()

    logger.info(f"Mean top-1 accuracy {acc_mean:.2f}% ± {acc_std:.2f}%")
    logger.info(f"Mean top-1 difference {acc_diff_mean:.2f}% ± {acc_diff_std:.2f}%")
    logger.info(f"Mean top-5 accuracy {top5_mean:.2f}% ± {top5_std:.2f}%")
    logger.info(f"Mean top-5 difference {top5_diff_mean:.2f}% ± {top5_diff_std:.2f}%")
