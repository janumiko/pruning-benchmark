cd pruning

python pruning_entry.py \
    wandb=test.yaml \
    pruner=unstructured \
    pruner.pruning_scheduler=one_shot \
    pruner.pruning_ratio=0.88 \
    pruner/pruning_config=resnet18_cifar/resnet18_cifar_unstructured_test.yaml \
    pruner.steps=1 \
    trainer=classification \
    trainer.early_stopper.patience=30 \
    trainer.early_stopper.override_epochs_to_inf=True \
    dataset=cifar10 \
    optimizer=sgd \
    optimizer.lr=0.001 \
    model.name=resnet18_cifar \
    model.checkpoint_path="checkpoints/resnet18_cifar10.pth"