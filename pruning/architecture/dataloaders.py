import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, random_split, DataLoader
from config.main_config import MainConfig


def get_cifar10(cfg: MainConfig) -> tuple[Dataset, Dataset, Dataset]:
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
            ),  # normalize the cifar10 images
        ]
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            test_transform,
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=cfg.dataset.path,
        train=True,
        download=cfg.dataset.download,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=cfg.dataset.path,
        train=False,
        download=cfg.dataset.download,
        transform=test_transform,
    )

    validate_dataset, test_dataset = random_split(test_dataset, [0.8, 0.2])

    return train_dataset, validate_dataset, test_dataset


def get_cifar100(cfg: MainConfig) -> tuple[Dataset, Dataset, Dataset]:
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025)),
        ]
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            test_transform,
        ]
    )

    train_dataset = datasets.CIFAR100(
        root=cfg.dataset.path,
        train=True,
        download=cfg.dataset.download,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR100(
        root=cfg.dataset.path,
        train=False,
        download=cfg.dataset.download,
        transform=test_transform,
    )

    validate_dataset, test_dataset = random_split(test_dataset, [0.8, 0.2])

    return train_dataset, validate_dataset, test_dataset


def get_dataset(
    cfg: MainConfig,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    match cfg.dataset.name.lower():
        case "cifar10":
            return get_cifar10(cfg)
        case "cifar100":
            return get_cifar100(cfg)
        case _:
            raise ValueError(f"Unknown dataset: {cfg.dataset.name}")


def get_dataloaders(
    cfg: MainConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset, validate_dataset, test_dataset = get_dataset(cfg)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataloaders.batch_size,
        shuffle=True,
        pin_memory=cfg.dataloaders.pin_memory,
        num_workers=cfg.dataloaders.num_workers,
    )
    validation_loader = DataLoader(
        validate_dataset,
        batch_size=cfg.dataloaders.batch_size,
        shuffle=False,
        pin_memory=cfg.dataloaders.pin_memory,
        num_workers=cfg.dataloaders.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.dataloaders.batch_size,
        shuffle=False,
        pin_memory=cfg.dataloaders.pin_memory,
        num_workers=cfg.dataloaders.num_workers,
    )

    return train_loader, validation_loader, test_loader
