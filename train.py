from logging import getLogger
import math

from datasets import load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

logger = getLogger(__name__)

# Load Dataset and Tokenizer
dataset = load_dataset("roneneldan/TinyStories", split="train")
val_dataset = load_dataset("roneneldan/TinyStories", split="validation")
# limit the dataset size
dataset = dataset.select(range(1000))
model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer.pad_token = tokenizer.eos_token


# Tokenize the Dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,  # Truncate if sequences are longer than max_length
        max_length=256,
    )


tokenized_datasets = dataset.map(
    tokenize_function, batched=True, num_proc=8, remove_columns=["text"]
)

# Split the tokenized dataset into training and validation sets
train_dataset, val_dataset = tokenized_datasets.train_test_split(test_size=0.1).values()

# Data collator to handle padding and create attention masks
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

dataloader = DataLoader(
    tokenized_datasets,
    batch_size=16,
    shuffle=True,
    collate_fn=data_collator,
    pin_memory=True,
    num_workers=4,
)

# DataLoader for the validation set
val_dataloader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=data_collator,
    pin_memory=True,
    num_workers=4,
)


# save the model checkpoint
torch.save(model.state_dict(), "tinystories1m.pth")

# Fine-Tune the Model
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
model.to(device)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        # Move batch to the correct device
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(dataloader)
    avg_perplexity = math.exp(avg_loss)
    print(
        f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Perplexity: {avg_perplexity:.4f}"
    )

    # Validation phase
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_dataloader)
    avg_val_perplexity = math.exp(avg_val_loss)
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Perplexity: {avg_val_perplexity:.4f}")

import torch.nn.utils.prune as prune

# get all the  linear layers


def get_parameters_to_prune(model: nn.Module, types_to_prune) -> list[tuple[nn.Module, str]]:
    """Get the parameters to prune from a model.

    Args:
        model (nn.Module): A PyTorch model.
        types_to_prune (Iterable[nn.Module]): Tuple of module types to prune. Ex. nn.Linear, nn.Conv2d.

    Returns:
        list[tuple[nn.Module, str]]: List of tuples containing the module and the parameter name.
    """
    return [
        (module, name)
        for module in model.modules()
        if isinstance(module, types_to_prune)
        for name, param in module.named_parameters()
        if param.requires_grad
    ]


# Prune the model
prune.global_unstructured(
    get_parameters_to_prune(model, (nn.Linear,)),
    pruning_method=prune.L1Unstructured,
    amount=0.2,
)

optimizer = optim.AdamW(model.parameters(), lr=5e-5)
model.to(device)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        # Move batch to the correct device
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(dataloader)
    avg_perplexity = math.exp(avg_loss)
    print(
        f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Perplexity: {avg_perplexity:.4f}"
    )

    # Validation phase
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_dataloader)
    avg_val_perplexity = math.exp(avg_val_loss)
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Perplexity: {avg_val_perplexity:.4f}")


print(f"Perplexity: {avg_perplexity:.4f}")
