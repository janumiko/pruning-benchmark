# Pruning Analysis

## Introduction
This repository contains code for analysis of various pruning methods.\
The code is written in PyTorch and supports distributed computing, has a CLI support which allows for easy execution from batch files.\
Configuration is flexible (but still can be expanded) and configuration sweep (Hydra multirun) is also possible.\
Experiments can be analyzed through robust logging using Hydra and Wandb.

## Installation
To install the required packages, run the following command:
```bash
conda env create -f environment.yml
```
Conda/Anaconda is required to run the above command. If you don't have it installed, you can install it from [here](https://www.anaconda.com/products/distribution).

## Usage
The pruning code is located in the [pruning directory](pruning).\
The entry point for the program is [pruning_entry.py](pruning/pruning_entry.py), the main file to run the pruning analysis. \
You can configure the code from CLI, or modify the configs in the [config directory](pruning/config).\
The config is using [Hydra](https://hydra.cc/), which is  a configuration system for Python apps.

To run a single pruning experiment you need to provide the necessary parameters which depend on the method/scheduler to use.\
You can check the configuration files in [config](pruning/config) directory for guidance.

## Models and datasets
You need to provide the model, dataset, path to dataset and checkpoint to use for the model.

- checkpoint path is provided through `_checkpoint_path` argument, for example: `_checkpoint_path=checkpoints/resnet18_cifar100.pth`.
- dataset path is provided through `dataset._path` argument.
- model is chosen through `model` argument. Supported models and the code for them is defined in [models directory](pruning/architecture/models).
- dataset is chosen through `dataset` argument. Supported datasets and the code for them is defined in [construct_dataset.py](pruning/architecture/construct_dataset.py).

## Configuration Overview

The configuration of the project is managed by Hydra and is divided into several parts, each corresponding to a Python file in the `config` directory:

#### `main_config.py`

This is the main configuration file, it defines the main configuration class `MainConfig` and registers it with Hydra.\
It also registers the configurations for optimizers, datasets, pruning schedulers, pruning methods, and metrics.

#### `methods.py`

Currently only magnitude methods are supported.
- `LnStructured` - structured pruning method, with the modifable norm and dimension which will be pruned.
- `GlobalL1Unstructured` - unstructured global pruning, prunes all weights according to l1 norm.

#### `optimizers.py`

This file defines the optimizers used in the project. The available optimizers are:

- `AdamW`
- `SGD`

You can specify the optimizer to use in the `optimizer` field of the main configuration.

#### `schedulers.py`

This file defines the pruning schedulers used in the project. The available schedulers are:\
Scheduler are objects which decide how the steps for pruning iterations are calculated.\
E.g. OneShotStepScheduler with step=0.6 will prune the 60% of the weights and do a one iteration of fine-tuning.

- `IterativeStepScheduler` 
- `OneShotStepScheduler`
- `LogarithmicStepScheduler`
- `ConstantStepScheduler`
- `ManualScheduler`

You can specify the scheduler to use in the `pruning.scheduler` field of the main configuration.\
The manual scheduler allows to provide user-provided values separately for every layer.\
They are passed through `pruning.scheduler.pruning_steps` argument.

#### `datasets.py`
Configuration for datasets like path, name and number of classes (for dynamic creation of models).\
Also, arguments `resize_value` and `crop_value` are provided, which allow to resize and crop the images.\
Common case for `Imagenet1k` is `dataset.resize_value=256` and `dataset.crop_value=224`.

Currently supported datasets include:
- `cifar10`
- `cifar100`
- `imagenet1k`

## Logging
Currently the logging is logged using Hydra, but [Wandb](https://wandb.ai/site) is supported and recommended.

## Examples

Structured manual pruning with 3 repeats and single pruning iteration and early stopping with patience for 10 epochs.
```bash
python pruning_entry.py model=resnet18_cifar dataset=cifar100 _repeat=3 optimizer=sgd optimizer.learning_rate=0.001 _checkpoint_path=checkpoints/resnet18_cifar100.pth pruning.scheduler=manual pruning.finetune_epochs=100 dataloaders.batch_size=256 pruning.method=ln_structured 'pruning.scheduler.pruning_steps=[[0.0, 0.2, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.0]]' pruning.method.norm=2 early_stopper.enabled=True early_stopper.patience=10 
```
