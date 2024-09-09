from config.constants import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    CIFAR100_MEAN,
    CIFAR100_STD,
    IMAGENET1K_MEAN,
    IMAGENET1K_STD,
)
from config.main_config import MainConfig
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from transformers import AutoTokenizer, DataCollatorForLanguageModeling


def get_cifar10(
    path: str, download: bool, resize_value: int | None = None, crop_value: int | None = None
) -> tuple[Dataset, Dataset]:
    """Constructs the CIFAR10 dataset.

    Args:
        path (str): Path to the dataset.
        download (bool): Should the dataset be downloaded.
        resize_value (int | None, optional): Value to resize the images to. Defaults to None.
        crop_value (int | None, optional): Value to crop the images to. Defaults to None.

    Returns:
        tuple[Dataset, Dataset, Dataset]: Tuple of train and test datasets.
    """
    common_transformations = []

    if resize_value is not None:
        common_transformations.append(
            transforms.Resize((resize_value, resize_value), antialias=True)
        )

    if crop_value is not None:
        common_transformations.append(transforms.CenterCrop(crop_value))

    common_transformations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )

    test_transform = transforms.Compose(common_transformations)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
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

    return train_dataset, validate_dataset, None


def get_cifar100(
    path: str, download: bool, resize_value: int | None = None, crop_value: int | None = None
) -> tuple[Dataset, Dataset]:
    """Constructs the CIFAR100 dataset.

    Args:
        path (str): Path to the dataset.
        download (bool): Should the dataset be downloaded.
        resize_value (int | None, optional): Value to resize the images to. Defaults to None.
        crop_value (int | None, optional): Value to crop the images to. Defaults to None.

    Returns:
        tuple[Dataset, Dataset]: Tuple of train and test datasets.
    """
    common_transformations = []

    if resize_value is not None:
        common_transformations.append(
            transforms.Resize((resize_value, resize_value), antialias=True)
        )

    if crop_value is not None:
        common_transformations.append(transforms.CenterCrop(crop_value))

    common_transformations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
        ]
    )

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

    return train_dataset, validate_dataset, None


def get_imagenet1k(
    path: str, resize_value: int | None = None, crop_value: int | None = None
) -> tuple[Dataset, Dataset]:
    """Constructs the ImageNet1K dataset.

    Args:
        path (str): Path to the dataset.
        resize_value (int | None): The size to resize the images to. Defaults to None.
        crop_value (int | None): The size to crop the images to. Defaults to None.

    Returns:
        tuple[Dataset, Dataset]: Train and validation datasets.
    """

    common_transformations = []

    if resize_value is not None:
        common_transformations.append(
            transforms.Resize((resize_value, resize_value), antialias=True)
        )

    if crop_value is not None:
        common_transformations.append(transforms.CenterCrop(crop_value))

    common_transformations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET1K_MEAN, std=IMAGENET1K_STD),
        ]
    )

    test_transform = transforms.Compose(common_transformations)

    train_transform = transforms.Compose(
        [
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

    return train_dataset, validate_dataset, None


def get_tinystories_gptneo(path: str, percent: int = 100) -> tuple[Dataset, Dataset]:
    """Get the TinyStories dataset tokenized with GPT-Neo.

    Args:
        path (str): Path to the cache for the dataset.
        percent (int, optional): Percentage of the original dataset to take. Defaults to 100.

    Returns:
        tuple[Dataset, Dataset]: _description_
    """
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token
    # Load Dataset and Tokenizer
    train_dataset = load_dataset("roneneldan/TinyStories", split="train", cache_dir=path)
    val_dataset = load_dataset("roneneldan/TinyStories", split="validation", cache_dir=path)
    # Tokenize the Dataset

    # take the percent of the dataset
    train_dataset = train_dataset.train_test_split(train_size=percent / 100, seed=42)["train"]
    val_dataset = val_dataset.train_test_split(train_size=percent / 100, seed=42)["train"]

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,  # Truncate if sequences are longer than max_length
            max_length=256,
        )

    tokenized_train = train_dataset.map(
        tokenize_function, batched=True, num_proc=32, remove_columns=["text"]
    )
    tokenized_val = val_dataset.map(
        tokenize_function, batched=True, num_proc=32, remove_columns=["text"]
    )

    # Data collator to handle padding and create attention masks
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    return tokenized_train, tokenized_val, data_collator


def get_dataset(
    cfg: MainConfig,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Get the dataset based on the provided configuration.

    Args:
        cfg (MainConfig): The main configuration object.

    Returns:
        tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]: A tuple containing the train and test datasets.
    """
    match cfg.dataset.name.lower():
        case "cifar10":
            return get_cifar10(
                cfg.dataset._path,
                cfg.dataset._download,
                resize_value=cfg.dataset.resize_value,
                crop_value=cfg.dataset.crop_value,
            )
        case "cifar100":
            return get_cifar100(
                cfg.dataset._path,
                cfg.dataset._download,
                resize_value=cfg.dataset.resize_value,
                crop_value=cfg.dataset.crop_value,
            )
        case "imagenet1k":
            return get_imagenet1k(
                cfg.dataset._path,
                resize_value=cfg.dataset.resize_value,
                crop_value=cfg.dataset.crop_value,
            )
        case "tiny_stories_gpt_neo":
            return get_tinystories_gptneo(cfg.dataset._path, cfg.dataset.percent)
        case _:
            raise ValueError(f"Unknown dataset: {cfg.dataset.name}")


def get_dataloaders(
    cfg: MainConfig,
) -> tuple[DataLoader, DataLoader]:
    """
    Get train and validation data loaders.

    Args:
        cfg (MainConfig): The main configuration object.

    Returns:
        tuple[DataLoader, DataLoader]: A tuple containing the train and validation data loaders.
    """
    train_dataset, validation_dataset, collate_fn = get_dataset(cfg)
    train_sampler = DistributedSampler(train_dataset) if cfg._gpus > 1 else None
    validation_sampler = DistributedSampler(validation_dataset) if cfg._gpus > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataloaders.batch_size,
        shuffle=(train_sampler is None),
        pin_memory=cfg.dataloaders._pin_memory,
        num_workers=cfg.dataloaders._num_workers,
        sampler=train_sampler,
        persistent_workers=cfg.dataloaders._persistent_workers,
        drop_last=cfg.dataloaders._drop_last,
        collate_fn=collate_fn,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.dataloaders.batch_size,
        shuffle=False,
        pin_memory=cfg.dataloaders._pin_memory,
        num_workers=cfg.dataloaders._num_workers,
        sampler=validation_sampler,
        persistent_workers=cfg.dataloaders._persistent_workers,
        collate_fn=collate_fn,
    )

    return train_loader, validation_loader
