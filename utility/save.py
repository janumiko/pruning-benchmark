import torch
import json

from torch import nn
from pathlib import Path


def save_model(model: nn.Module, path: str, filename: str) -> None:
    """Save model into given path

    Args:
        model (nn.Module): model to be saved.
        path (str): path where the model will be saved.
    """

    torch.save(model.state_dict(), Path(path) / Path(filename))


def save_model_with_metadata(
    model: nn.Module, path: str, model_name: str, metadata: dict, create: bool = True
) -> None:
    """Save model with metadata json to models directory

    Args:
        model (nn.Module): the model to be saved
        name (str): name of the model
        metadata (dict): a dictionary containing metadata of the model
        create (bool, optional): create the directory if not exists. Defaults to True.
    """

    target_path = Path(path)
    if create:
        target_path.mkdir(parents=True, exist_ok=True)

    (target_path / Path("metadata.json")).write_text(json.dumps(metadata))
    torch.save(model.state_dict(), target_path / Path(f"{model_name}.pth"))
