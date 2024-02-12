# Pruning Analysis

## Introduction
This repository contains code for analysis of various pruning methods.
The code is written in PyTorch, including the built-in pruning methods.

### Installation
To install the required packages, run the following command:
```bash
conda env create -f environment.yml
```
Conda/Anaconda is required to run the above command. If you don't have it installed, you can install it from [here](https://www.anaconda.com/products/distribution).

### Usage
The pruning code is located in the `pruning` directory.
The entry point for the program is `pruning_loop.py`, the main file to run the pruning analysis. 
You can configure the code from CLI, or modyfy the configs in `conf` directory. The config is using [Hydra](https://hydra.cc/), which is  a configuration system for Python apps.

To run a single pruning you need to provide `pruning.iterations`, `pruning.finetune_epochs` and `pruning.iteration_rate` parameters. For example:
```bash
python pruning_entry.py pruning.iterations=3 pruning.finetune_epochs=1 pruning.iteration_rate=0.02
```

You can do a multi-run by using following hydra syntax:
```bash
python pruning_entry.py --multirun pruning.iterations=1,2,3 pruning.finetune_epochs=1 pruning.iteration_rate=0.01,0.02
```