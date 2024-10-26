#!/bin/bash -l
#SBATCH -J cifar
#SBATCH --mem=100G
#SBATCH --time=48:00:00 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH -A plgpruning-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH -C memfs

module add CUDA/12.0.0
module add Miniconda3/23.3.1-0
module add GCC/10.3.0
module load Ninja/1.10.2
export CUDA_HOME=/net/software/v1/software/CUDA/12.0.0

nvidia-smi -L
source activate /net/tscratch/people/plgkowalski1024/pruning_env

cd ../pruning
python pruning_entry.py \
    paths=athena.yaml \
    pruner=structured \
    pruner.importance=norm_importance \
    pruner.pruning_scheduler=$1 \
    pruner/pruning_config=$2 \
    pruner.steps=$3 \
    trainer.early_stopper.enabled=$4 \
    trainer.early_stopper.patience=$5 \
    trainer.early_stopper.override_epochs_to_inf=True \
    trainer=classification \
    trainer.epochs=$6 \
    dataset=$7 \
    optimizer=sgd \
    optimizer.lr=0.001 \
    model.name=$8 \
    model.checkpoint_path=$9 \
    train_dataloader.batch_size=512 \
    validation_dataloader.batch_size=512 \
    wandb=default.yaml \
    wandb.job_type=${10} \
    seed=${11} \


#    distributed.enabled=True \
#    distributed.world_size=2 \
#    distributed.init_method="file://${MEMFS}/ddp_init_${SLURM_JOB_ID}.txt" \

