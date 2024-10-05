cd pruning

python pruning_entry.py \
    pruner=structured_magnitude \
    pruner.importance=norm_importance \
    pruner.pruning_scheduler=one_shot \
    pruner/pruning_config=resnet18_cifar/resnet18_cifar_test.yaml \
    trainer=classification \
    trainer.early_stopper.enabled=True \
    trainer.early_stopper.patience=5 \
    dataset=cifar10 \
    optimizer=sgd \
    model.name=resnet18_cifar \
    model.checkpoint_path="checkpoints/resnet18_cifar10.pth"