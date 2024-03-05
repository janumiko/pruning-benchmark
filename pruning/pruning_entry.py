import hydra
import torch
import torch.nn.utils.prune as prune
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from architecture.dataloaders import get_dataloaders
from architecture.construct_model import construct_model
from architecture.pruning_loop import prune_model
import architecture.utility as utility
import datetime
import wandb
import logging
import numpy as np

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """Main function for the pruning entry point

    Args:
        cfg (DictConfig): Hydra config object with all the settings. (Located in conf/config.yaml)
    """

    logger.info(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Save the model to the Hydra output directory
    hydra_output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    if cfg.seed.is_set:
        utility.training.set_reproducibility(cfg.seed.value)

    base_model: torch.nn.Module = construct_model(cfg).to(device)
    train_dl, valid_dl, test_dl = get_dataloaders(cfg)
    cross_entropy = torch.nn.CrossEntropyLoss()
    base_test_loss, base_test_accuracy = utility.training.test(
        module=base_model,
        test_dl=test_dl,
        loss_function=cross_entropy,
        device=device,
    )
    logger.info(
        f"Base test loss: {base_test_loss:.4f} base test accuracy: {base_test_accuracy:.2f}%"
    )

    test_results = []
    for i in range(cfg.repeat):
        logger.info(f"Repeat {i+1}/{cfg.repeat}")

        model = construct_model(cfg).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
        )
        pruning_parameters = utility.pruning.get_parameters_to_prune(model)
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
        )

        test_loss, test_accuracy = utility.training.test(
            module=model,
            test_dl=test_dl,
            loss_function=cross_entropy,
            device=device,
        )
        test_results.append(test_accuracy)
        logger.info(f"Test loss: {test_loss:.4f} Test accuracy: {test_accuracy:.2f}%")

        current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        torch.save(
            model.state_dict(),
            hydra_output_dir / f"{cfg.model.name}_{current_date}.pth",
        )

    test_results = np.array(test_results)
    logger.info(f"Base test accuracy: {base_test_accuracy:.2f}%")
    logger.info(f"Average test accuracy: {test_results.mean():.2f}%")
    logger.info(f"Standard deviation: {test_results.std():.2f} percentage points")
    logger.info(
        f"Average accuracy loss after pruning: {base_test_accuracy - test_results.mean():.2f} percentage points"
    )


if __name__ == "__main__":
    main()
