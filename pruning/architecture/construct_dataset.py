from config.main_config import MainConfig
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from config.constants import (
    CIFAR100_MEAN,
    CIFAR100_STD,
    IMAGENET1K_MEAN,
    IMAGENET1K_STD,
    CIFAR10_MEAN,
    CIFAR10_STD,
)


def get_cifar10(
    path: str, download: bool, resize_value: int | None = None
) -> tuple[Dataset, Dataset]:
    """Constructs the CIFAR10 dataset.

    Args:
        path (str): Path to the dataset.
        download (bool): Should the dataset be downloaded.
        resize_value (int | None, optional): Value to resize the images to. Defaults to None.

    Returns:
        tuple[Dataset, Dataset, Dataset]: Tuple of train and test datasets.
    """

    common_transformations = [
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ]

    if resize_value is not None:
        common_transformations.insert(0, transforms.Resize((resize_value, resize_value)))

    test_transform = transforms.Compose(common_transformations)

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


def get_cifar100(
    path: str, download: bool, resize_value: int | None = None
) -> tuple[Dataset, Dataset]:
    """Constructs the CIFAR100 dataset.

    Args:
        path (str): Path to the dataset.
        download (bool): Should the dataset be downloaded.
        resize_value (int | None, optional): Value to resize the images to. Defaults to None.

    Returns:
        tuple[Dataset, Dataset]: Tuple of train and test datasets.
    """

    common_transformations = [
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
    ]

    if resize_value is not None:
        common_transformations.insert(0, transforms.Resize((resize_value, resize_value)))

    test_transform = transforms.Compose(common_transformations)

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


def get_imagenet1k(path: str, resize_value: int | None = None) -> tuple[Dataset, Dataset]:
    """Constructs the ImageNet1K dataset.

    Args:
        path (str): Path to the dataset.
        resize_value (int | None): The size to resize the images to. Defaults to None.

    Returns:
        tuple[Dataset, Dataset]: Train and validation datasets.
    """

    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET1K_MEAN, std=IMAGENET1K_STD),
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
        transform=train_transform,
    )

    validate_dataset = datasets.ImageNet(
        root=path,
        split="val",
        transform=test_transform,
    )

    return train_dataset, validate_dataset


def get_dataset(
    cfg: MainConfig,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    match cfg.dataset.name.lower():
        case "cifar10":
            return get_cifar10(
                cfg.dataset._path, cfg.dataset._download, resize_value=cfg.dataset.resize_value
            )
        case "cifar100":
            return get_cifar100(
                cfg.dataset._path, cfg.dataset._download, resize_value=cfg.dataset.resize_value
            )
        case "imagenet1k":
            return get_imagenet1k(cfg.dataset._path, resize_value=cfg.dataset.resize_value)
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
