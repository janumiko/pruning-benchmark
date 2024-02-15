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


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model: torch.nn.Module = construct_model(cfg).to(device)
    train_dl, valid_dl, test_dl = get_dataloaders(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )
    cross_entropy = torch.nn.CrossEntropyLoss()
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

    pruned_model = prune_model(
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

    test_loss, test_accuracy = utility.training.validate(
        module=pruned_model,
        valid_dl=test_dl,
        loss_function=cross_entropy,
        device=device,
    )

    print(f"Test accuracy: {test_accuracy:.2f}%")

    # Save the model to the Hydra output directory
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    torch.save(
        pruned_model.state_dict(), output_dir / f"{cfg.model.name}_{current_date}.pth"
    )


if __name__ == "__main__":
    main()
