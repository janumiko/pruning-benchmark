import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, random_split, DataLoader
from omegaconf import DictConfig


def get_cifar10(cfg: DictConfig) -> tuple[Dataset, Dataset, Dataset]:
    normalize_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
            ),  # normalize the cifar10 images
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=cfg.dataset.path,
        train=True,
        download=cfg.dataset.download,
        transform=normalize_tensor,
    )
    train_dataset, validate_dataset = random_split(train_dataset, [0.8, 0.2])
    test_dataset = datasets.CIFAR10(
        root=cfg.dataset.path,
        train=False,
        download=cfg.dataset.download,
        transform=normalize_tensor,
    )

    return train_dataset, validate_dataset, test_dataset


def get_dataset(
    cfg: DictConfig,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    if cfg.dataset.name.lower() == "cifar10":
        return get_cifar10(cfg)
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset.name}")


def get_dataloaders(
    cfg: DictConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset, validate_dataset, test_dataset = get_dataset(cfg)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.pruning.batch_size, shuffle=True
    )
    validation_loader = DataLoader(
        validate_dataset, batch_size=cfg.pruning.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.pruning.batch_size, shuffle=False
    )

    return train_loader, validation_loader, test_loader
