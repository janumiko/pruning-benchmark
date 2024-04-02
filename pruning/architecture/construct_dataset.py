from config.main_config import MainConfig
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def get_cifar10(path: str, download: bool) -> tuple[Dataset, Dataset, Dataset]:
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
        root=path,
        train=True,
        download=download,
        transform=train_transform,
    )

    validate_dataset = datasets.CIFAR10(
        root=path,
        train=False,
        download=download,
        transform=test_transform,
    )

    return train_dataset, validate_dataset


def get_cifar100(path: str, download: bool) -> tuple[Dataset, Dataset]:
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
        root=path,
        train=True,
        download=download,
        transform=train_transform,
    )

    validate_dataset = datasets.CIFAR100(
        root=path,
        train=False,
        download=download,
        transform=test_transform,
    )

    return train_dataset, validate_dataset


def get_imagenet1k(path: str, download: bool) -> tuple[Dataset, Dataset]:
    """Constructs the ImageNet1K dataset.

    Args:
        path (str): Path to the dataset.
        download (bool): Whether to download the dataset.

    Returns:
        tuple[Dataset, Dataset]: Train and validation datasets.
    """

    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            test_transform,
        ]
    )

    train_dataset = datasets.ImageNet(
        root=path,
        split="train",
        download=download,
        transform=train_transform,
    )

    validate_dataset = datasets.ImageNet(
        root=path,
        split="val",
        download=download,
        transform=test_transform,
    )

    return train_dataset, validate_dataset


def get_dataset(
    cfg: MainConfig,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    match cfg.dataset.name.lower():
        case "cifar10":
            return get_cifar10(cfg.dataset._path, cfg.dataset._download)
        case "cifar100":
            return get_cifar100(cfg.dataset._path, cfg.dataset._download)
        case "imagenet1k":
            return get_imagenet1k(cfg.dataset._path, cfg.dataset._download)
        case _:
            raise ValueError(f"Unknown dataset: {cfg.dataset.name}")


def get_dataloaders(
    cfg: MainConfig,
) -> tuple[DataLoader, DataLoader]:
    train_dataset, validate_dataset = get_dataset(cfg)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataloaders.batch_size,
        shuffle=True,
        pin_memory=cfg.dataloaders._pin_memory,
        num_workers=cfg.dataloaders._num_workers,
    )
    validation_loader = DataLoader(
        validate_dataset,
        batch_size=cfg.dataloaders.batch_size,
        shuffle=False,
        pin_memory=cfg.dataloaders._pin_memory,
        num_workers=cfg.dataloaders._num_workers,
    )

    return train_loader, validation_loader
