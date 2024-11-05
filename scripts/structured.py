import subprocess


def run_bash_command_no_wait(command):
    try:
        # Start the command and capture the output
        process = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            print(f"Started command: {command}")
            print(f"Command output: {stdout.strip()}")
            job_id = stdout.strip().split()[-1]
            return job_id
        else:
            print(f"Failed to start command '{command}' with error: {stderr.strip()}")
    except Exception as e:
        print(f"Failed to start command '{command}' with error: {e}")


def multirun(command: str, run_num: int):
    job_id = None
    for i in range(run_num):
        if i == 0:
            command_copy = f"sbatch {command} {i}"
        else:
            command_copy = f"sbatch --dependency=afterok:{job_id} {command} {i}"
        job_id = run_bash_command_no_wait(command_copy)


def submit_jobs(
    pruning_configs,
    scheduler_name,
    steps_list,
    patience_list,
    epochs_list,
    job_type,
    dataset,
    model,
    checkpoint,
    run_num=1,
):
    for config in pruning_configs:
        for steps in steps_list:
            for patience in patience_list:
                command = (
                    f"structured_run.sh {scheduler_name} {config} {steps} True {patience} "
                    f"100000 {dataset} {model} {checkpoint} {job_type}"
                )
                multirun(command, run_num)
            for epochs in epochs_list:
                command = (
                    f"structured_run.sh {scheduler_name} {config} {steps} False 0 "
                    f"{epochs} {dataset} {model} {checkpoint} {job_type}"
                )
                multirun(command, run_num)


def main():
    model = "resnet18_cifar"
    checkpoint = "/net/pr2/projects/plgrid/plggpruning/models/resnet18_cifar100.pth"
    dataset = "cifar100"
    run_num = 3

    one_shot_flag = True
    geometric_flag = False
    constant_flag = False

    pruning_configs = [
        "resnet18_cifar/structured/pruning_0.2-0.5.yaml",
        # "resnet18_cifar/structured/pruning_0.4-0.7.yaml",
        "resnet18_cifar/structured/pruning_0.5-0.8.yaml",
        # "resnet18_cifar/structured/pruning_0.65-0.95.yaml",
    ]

    if one_shot_flag:
        scheduler_name = "one_shot"
        steps_list = [1]
        patience_list = [10, 20, 50, 100, 500, 1000]
        epochs_list = [100, 500, 1000, 2000]
        job_type = f"structured_{scheduler_name}_25-10-2024"

        submit_jobs(
            pruning_configs,
            scheduler_name,
            steps_list,
            patience_list,
            epochs_list,
            job_type,
            dataset,
            model,
            checkpoint,
            run_num
        )

    if geometric_flag:
        scheduler_name = "geometric"
        steps_list = [50, 25, 10, 5]
        patience_list = [10, 20, 50]
        epochs_list = [20, 50, 100]
        job_type = f"structured_{scheduler_name}_25-10-2024"

        submit_jobs(
            pruning_configs,
            scheduler_name,
            steps_list,
            patience_list,
            epochs_list,
            job_type,
            dataset,
            model,
            checkpoint,
            run_num
        )

    if constant_flag:
        scheduler_name = "constant"
        steps_list = [50, 25, 10, 5]
        patience_list = [10, 20, 50]
        epochs_list = [20, 50, 100]
        job_type = f"structured_{scheduler_name}_25-10-2024"

        submit_jobs(
            pruning_configs,
            scheduler_name,
            steps_list,
            patience_list,
            epochs_list,
            job_type,
            dataset,
            model,
            checkpoint,
            run_num
        )


if __name__ == "__main__":
    main()