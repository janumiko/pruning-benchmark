cd pruning
python pruning_entry.py trainer=classification trainer.early_stopper.patience=5 dataset=cifar10 optimizer=sgd model.name=resnet18_cifar model.checkpoint_path="checkpoints/resnet18_cifar10.pth"