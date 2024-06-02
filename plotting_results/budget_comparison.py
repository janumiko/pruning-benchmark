import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

def find_newest_folder_date_time(directory, prefix):
    # List all the folders in the given directory
    folders = [f for f in os.listdir(directory) if f.startswith(prefix)]

    # Initialize variables to keep track of the newest folder date and time
    newest_folder_date_time = None
    newest_datetime = datetime.datetime.min

    # Loop through each folder
    for folder in folders:
        # Extract the date and time part of the folder name
        try:
            date_time_str = folder.split('_', 2)[1] + '_' + folder.split('_', 2)[2]
            folder_datetime = datetime.datetime.strptime(date_time_str, '%Y-%m-%d_%H-%M-%S')

            # Update the newest folder date and time if this one is more recent
            if folder_datetime > newest_datetime and folder.split('_', 2)[0]==prefix:
                newest_datetime = folder_datetime
                newest_folder_date_time = date_time_str
        except (IndexError, ValueError):
            # Handle cases where the folder name format is incorrect
            continue
    print(newest_folder_date_time)
    return newest_folder_date_time

    # return newest_folder

dataset="cifar100"
d = find_newest_folder_date_time("../csvs", dataset)
# date="_2024-05-15_11-25-32"
# dataset="cifar100"
date="_"+d

# Load the datasets
data_one_shot = pd.read_csv(f"../csvs/{dataset}{date}/pruning_results_{dataset}_one-shot{date}.csv")
data_iterative = pd.read_csv(f"../csvs/{dataset}{date}/pruning_results_{dataset}_iterative{date}.csv")
data_constant = pd.read_csv(f"../csvs/{dataset}{date}/pruning_results_{dataset}_constant{date}.csv")

# Function to filter data based on pruning percentage, learning rate, and strategy-specific conditions
def filter_data(data, pruning_percentages, learning_rate, finetune_epochs=None, scheduler_step=None, early_stopper_patience=None, early_stopper_it_enabled=False, name=None):
    condition = (
        (data['optimizer.learning_rate'] == learning_rate) &
        (data['pruned_precent'].isin(pruning_percentages))
    )
    if finetune_epochs is not None:
        condition &= (data['pruning.finetune_epochs'] == finetune_epochs)
    if scheduler_step is not None:
        condition &= (np.round(data['pruning.scheduler.step'], 2) == scheduler_step)
    if name in ["iterative", "constant"]:
        condition &= data["early_stopper.enabled"] == early_stopper_it_enabled
    if early_stopper_patience is not None and early_stopper_it_enabled:
        condition &= data["early_stopper.patience"] == early_stopper_patience
    return data[condition]

# Function to find points close to specified epoch intervals
def find_points_close_to_epochs(data, start, end, step):
    selected_points = []
    for epoch_target in range(start, end + step, step):
        closest_row = data.iloc[(data['total_epoch_mean'] - epoch_target).abs().argsort()[:1]]
        selected_points.append(closest_row)
    return pd.concat(selected_points)

# Filtering data based on a specific pruning percentage and other parameters
pruning_percentage_of_interest = [70]
lr = 0.001
early_stopper_it_enabled = False
lr=0.001; iter_step_it=None; iter_step_const=None; iter_ft=None; iter_early_stopper_patience=None; early_stopper_it_enabled=True

filtered_one_shot = filter_data(data_one_shot, pruning_percentage_of_interest, lr, name="oneshot")
filtered_iterative = filter_data(data_iterative, pruning_percentage_of_interest, lr, name="iterative", scheduler_step=iter_step_it, finetune_epochs=iter_ft, early_stopper_patience=iter_early_stopper_patience, early_stopper_it_enabled=early_stopper_it_enabled)
filtered_constant = filter_data(data_constant, pruning_percentage_of_interest, lr, name="constant", finetune_epochs=iter_ft, scheduler_step=iter_step_const, early_stopper_patience=iter_early_stopper_patience, early_stopper_it_enabled=early_stopper_it_enabled)

# Select points close to specified epoch intervals for each strategy
epoch_start = 0
epoch_end = 200
epoch_step = 20
selected_one_shot = find_points_close_to_epochs(filtered_one_shot, epoch_start, epoch_end, epoch_step)
selected_iterative = find_points_close_to_epochs(filtered_iterative, epoch_start, epoch_end, epoch_step)
selected_constant = find_points_close_to_epochs(filtered_constant, epoch_start, epoch_end, epoch_step)

# Plotting the selected pointsyou can run this code for both cifar10 and cifar100
plt.figure(figsize=(12, 8))
plt.scatter(selected_one_shot['total_epoch_mean'], selected_one_shot['top1_accuracy_mean'], color='red', marker='o', s=120, label='One-Shot')
plt.scatter(selected_iterative['total_epoch_mean'], selected_iterative['top1_accuracy_mean'], color='green', marker='o', s=120, label='Iterative')
plt.scatter(selected_constant['total_epoch_mean'], selected_constant['top1_accuracy_mean'], color='blue', marker='o', s=120, label='Constant')
plt.xlabel('Total Epoch Mean')
plt.ylabel('Top-1 Accuracy Mean')
plt.title('Accuracy vs Epochs for Different Pruning Strategies')
plt.legend()
plt.grid(True)
plt.savefig(f"plots/budget_vs_acc/{dataset}_pruingrate_{pruning_percentage_of_interest}_three_lr{lr}_earlystopper_{early_stopper_it_enabled}_iterstepit{iter_step_it}_iterstepconst{iter_step_const}_iterft{iter_ft}.png")
plt.show()
