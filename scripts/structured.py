import subprocess


def run_bash_command_no_wait(command):
    try:
        # Start the command without waiting for it to finish
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"Started command: {command}")
        print(f"Process ID: {process.pid}")
    except Exception as e:
        print(f"Failed to start command '{command}' with error: {e}")


model = "resnet18_cifar"
checkpoint = f"/net/pr2/projects/plgrid/plggpruning/models/resnet18_cifar100.pth"
dataset = "cifar100"
seed = 0

one_shot_flag = True

################# one-shot #################
if one_shot_flag:
    scheluder = "one_shot"
    steps = 1
    pruninig_config = [
        "resnet18_cifar/structured/pruning_0.4-0.7.yaml", 
        "resnet18_cifar/structured/pruning_0.5-0.8.yaml", 
        "resnet18_cifar/structured/pruning_0.6-0.9.yaml", 
        "resnet18_cifar/structured/pruning_0.65-0.95.yaml"
    ]

    patience = [10, 20, 50, 100, 500, 1000]
    epochs = [100, 500, 1000, 2000]

    job_type = f"structured_{scheluder}_25-10-2024"

    for config in pruninig_config:
        for p in patience:
            command = f"sbatch structured_run.sh {scheluder} {config} {steps} True {p} {100_000} {dataset} {model} {checkpoint} {job_type} {seed}"
            run_bash_command_no_wait(command)
        for e in epochs:
            command = f"sbatch structured_run.sh {scheluder} {config} {steps} False 0 {e} {dataset} {model} {checkpoint} {job_type} {seed}"
            run_bash_command_no_wait(command)
