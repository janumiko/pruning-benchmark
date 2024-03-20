import datetime
import logging
from pathlib import Path

from architecture.construct_model import construct_model
from architecture.construct_optimizer import construct_optimizer
from architecture.dataloaders import get_dataloaders
from architecture.pruning_loop import prune_model
import architecture.utility as utility
from architecture.utility.summary import strip_underscore_keys
from config.main_config import MainConfig
import hydra
from omegaconf import OmegaConf
import pandas as pd
import torch
from torch import nn
import torch.nn.utils.prune as prune
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
    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = (
        f"{cfg.model.name}_"
        f"{cfg.dataset.name}_"
        f"{cfg.pruning.iterations}_"
        f"{cfg.pruning.iteration_rate}_"
        f"{cfg.pruning.finetune_epochs}_"
        f"{current_date}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model: nn.Module = construct_model(cfg).to(device)
    train_dl, valid_dl, test_dl = get_dataloaders(cfg)
    cross_entropy = nn.CrossEntropyLoss()
    base_test_loss, base_test_accuracy, base_test_top5acc = utility.training.test(
        module=base_model,
        test_dl=test_dl,
        loss_function=cross_entropy,
        device=device,
    )
    logger.info(f"Base test.loss: {base_test_loss:.4f}")
    logger.info(f"Base top-1 accuracy: {base_test_accuracy:.2f}%")
    logger.info(f"Base top-5 accuracy: {base_test_top5acc:.2f}%")

    if cfg._seed.is_set:
        utility.training.set_reproducibility(cfg._seed.value)

    early_stopper = None
    if cfg.pruning.early_stopping:
        early_stopper = utility.training.EarlyStopper(
            patience=cfg.early_stopper.patience,
            min_delta=cfg.early_stopper.min_delta,
        )

    results = []
    for i in range(cfg._repeat):
        wandb_run = None
        if cfg._wandb.logging:
            dict_config = OmegaConf.to_container(cfg, resolve=True)
            dict_config = strip_underscore_keys(dict_config)

            wandb_run = wandb.init(
                project=cfg._wandb.project, group=run_name, name=f"run_{i+1}/{cfg._repeat}", config=dict_config, job_type="test", entity="KowalskiTeam"
            )

        logger.info(f"Repeat {i+1}/{cfg._repeat}")

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
                "loss": test_loss,
                "top-1 accuracy": test_accuracy,
                "top-5 accuracy": test_top5acc,
                "top-1 difference": base_test_accuracy - test_accuracy,
                "top-5 difference": base_test_top5acc - test_top5acc,
            }
        )

        if wandb_run:
            row = results[-1].copy()
            row.update({
                "base_loss": base_test_loss,
                "base_top-1_accuracy": base_test_accuracy,
                "base_top-5_accuracy": base_test_top5acc,
            })
            table = wandb.Table(data=[list(row.values())], columns=list(row.keys()))
            wandb_run.log({"pruning_results": table})
            wandb_run.finish()

        if cfg._save_checkpoints:
            current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            torch.save(
                model.state_dict(),
                hydra_output_dir / f"{cfg.model.name}_{i}_{current_date}.pth",
            )

    results_df = pd.DataFrame(results).round(decimals=2)
    utility.summary.log_summary(results_df)
    utility.summary.save_results_with_config(
        results_df, cfg, hydra_output_dir / f"{current_date}.csv"
    )


if __name__ == "__main__":
    main()
