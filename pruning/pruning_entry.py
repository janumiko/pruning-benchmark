import hydra
import torch
import torch.nn.utils.prune as prune
from torch import nn
from pathlib import Path
from omegaconf import OmegaConf
from architecture.dataloaders import get_dataloaders
from architecture.construct_model import construct_model
from architecture.pruning_loop import prune_model
from architecture.utility.summary import log_summary
from architecture.construct_optimizer import construct_optimizer
import architecture.utility as utility
import datetime
import wandb
import logging
import pandas as pd
from config.main_config import MainConfig

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main_config", version_base="1.2")
def main(cfg: MainConfig) -> None:
    """Main function for the pruning entry point

    Args:
        cfg (MainConfig): Hydra config object with all the settings. (Located in config/main_config.py)
    """

    logger.info(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Save the model to the Hydra output directory
    hydra_output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )
    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger.info(f"Hydra output directory: {hydra_output_dir}")

    base_model: nn.Module = construct_model(cfg).to(device)
    train_dl, valid_dl, test_dl = get_dataloaders(cfg)
    cross_entropy = nn.CrossEntropyLoss()
    base_test_loss, base_test_accuracy, base_test_top5acc = utility.training.test(
        module=base_model,
        test_dl=test_dl,
        loss_function=cross_entropy,
        device=device,
    )
    logger.info(f"Base test loss: {base_test_loss:.4f}")
    logger.info(f"Base top-1 Accuracy: {base_test_accuracy:.2f}%")
    logger.info(f"Base top-5 accuracy: {base_test_top5acc:.2f}%")

    if cfg.seed.is_set:
        utility.training.set_reproducibility(cfg.seed.value)

    early_stopper = None
    if cfg.pruning.early_stopping:
        early_stopper = utility.training.EarlyStopper(
            patience=cfg.early_stopper.patience,
            min_delta=cfg.early_stopper.min_delta,
        )

    results = []
    for i in range(cfg.repeat):
        logger.info(f"Repeat {i+1}/{cfg.repeat}")

        model = construct_model(cfg).to(device)
        optimizer = construct_optimizer(cfg, model)
        pruning_parameters = utility.pruning.get_parameters_to_prune(
            model, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)
        )
        pruning_amount = int(
            round(
                utility.pruning.calculate_parameters_amount(pruning_parameters)
                * cfg.pruning.iteration_rate
            )
        )

        wandb_run = None
        if cfg.wandb.logging:
            wandb_run = wandb.init(project=cfg.wandb.project)

        prune_model(
            model=model,
            method=prune.L1Unstructured,
            parameters_to_prune=pruning_parameters,
            optimizer=optimizer,
            loss_fn=cross_entropy,
            iterations=cfg.pruning.iterations,
            finetune_epochs=cfg.pruning.finetune_epochs,
            pruning_amount=pruning_amount,
            train_dl=train_dl,
            valid_dl=valid_dl,
            device=device,
            wandb_run=wandb_run,
            early_stopper=early_stopper,
        )

        test_loss, test_accuracy, test_top5acc = utility.training.test(
            module=model,
            test_dl=test_dl,
            loss_function=cross_entropy,
            device=device,
        )

        results.append(
            {
                "Test loss": test_loss,
                "Top-1 accuracy": test_accuracy,
                "Top-5 accuracy": test_top5acc,
                "Top-1 difference": base_test_accuracy - test_accuracy,
                "Top-5 difference": base_test_top5acc - test_top5acc,
            }
        )

        if cfg.save_checkpoints:
            current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            torch.save(
                model.state_dict(),
                hydra_output_dir / f"{cfg.model.name}_{i}_{current_date}.pth",
            )

    results_df = pd.DataFrame(results).round(2)
    log_summary(results_df)

    results_df["Model"] = cfg.model.name
    results_df["Dataset"] = cfg.dataset.name
    results_df["Total pruning percentage"] = (
        cfg.pruning.iterations * cfg.pruning.iteration_rate
    )
    results_df["Finetune epochs"] = cfg.pruning.finetune_epochs
    results_df["Early stopping"] = cfg.pruning.early_stopping
    results_df.to_csv(
        hydra_output_dir / f"{current_date}.csv",
        mode="w",
        header=True,
        index=True,
        float_format="%.2f",
    )


if __name__ == "__main__":
    main()
