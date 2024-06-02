import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os


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

dataset="cifar10"
d = find_newest_folder_date_time("../csvs", dataset)
# date="_2024-05-15_11-25-32"
# dataset="cifar100"
date="_"+d

# Load the datasets
data_one_shot = pd.read_csv(f"../csvs/{dataset}{date}/pruning_results_{dataset}_one-shot{date}.csv")
data_iterative = pd.read_csv(f"../csvs/{dataset}{date}/pruning_results_{dataset}_iterative{date}.csv")
data_constant = pd.read_csv(f"../csvs/{dataset}{date}/pruning_results_{dataset}_constant{date}.csv")

# Function to filter and process data based on learning rate and pruning percentages
# Function to filter and process data based on learning rate, pruning percentages, and specific conditions for iterative strategy
# Function to filter and process data based on learning rate, pruning percentages, and additional conditions
def filter_data(data, pruning_percentages, learning_rate, finetune_epochs=None, scheduler_step=None, early_stopper_patience=None, early_stopper_it_enabled=False, name=None):
    condition = (
        (data['optimizer.learning_rate'] == learning_rate) &
        (data['pruned_precent'].isin(pruning_percentages))
    )
    if finetune_epochs is not None:
        condition &= (data['pruning.finetune_epochs'] == finetune_epochs)
    if scheduler_step is not None:
        condition &= (np.round(data['pruning.scheduler.step'], 2) == scheduler_step)
    if name=="iterative" or name=="constant":
        condition &= data["early_stopper.enabled"] == early_stopper_it_enabled
    if early_stopper_patience is not None and early_stopper_it_enabled:
        condition &= data["early_stopper.patience"] == early_stopper_patience
    return data[condition]


# Define the pruning percentages of interest
pruning_percentages = [70, 80, 88, 92, 96]

# Filter data ------------------------------------------
lr=0.001; iter_step_it=None; iter_step_const=None; iter_ft=None; iter_early_stopper_patience=None; early_stopper_it_enabled=True
filtered_one_shot = filter_data(data_one_shot, pruning_percentages, 0.001, name="oneshot")
filtered_iterative = filter_data(data_iterative, pruning_percentages, learning_rate=lr, name="iterative", scheduler_step=iter_step_it, finetune_epochs=iter_ft, early_stopper_patience=iter_early_stopper_patience, early_stopper_it_enabled=early_stopper_it_enabled)
filtered_constant = filter_data(data_constant, pruning_percentages, 0.001, name="constant", finetune_epochs=iter_ft, scheduler_step=iter_step_const, early_stopper_patience=iter_early_stopper_patience, early_stopper_it_enabled=early_stopper_it_enabled)

# For one-shot, select entries with the best accuracy for each pruning percentage
filtered_one_shot = filtered_one_shot.loc[filtered_one_shot.groupby('pruned_precent')['top1_accuracy_mean'].idxmax()]
filtered_iterative = filtered_iterative.loc[filtered_iterative.groupby('pruned_precent')['top1_accuracy_mean'].idxmax()]
filtered_constant = filtered_constant.loc[filtered_constant.groupby('pruned_precent')['top1_accuracy_mean'].idxmax()]


# Plotting the accuracies
plt.figure(figsize=(10, 6))
plt.plot(filtered_one_shot['pruned_precent'], filtered_one_shot['top1_accuracy_mean'], label='One-Shot', marker='o')
plt.plot(filtered_iterative['pruned_precent'], filtered_iterative['top1_accuracy_mean'], label='Iterative', marker='s')
plt.plot(filtered_constant['pruned_precent'], filtered_constant['top1_accuracy_mean'], label='Constant', marker='^')

plt.xlabel('Pruning Percentage')
plt.ylabel('Top-1 Accuracy Mean')
plt.title(f'{dataset}: Comparison of Top-1 Accuracies by Pruning Strategy for \n lr {lr}, early stopper {early_stopper_it_enabled}, iter_step iter {iter_step_it}, iter_step const {iter_step_const}, iter_early_stopper_patience {iter_early_stopper_patience}, iter_finetune {iter_ft}')
plt.legend()
plt.grid(True)
plt.savefig(f"plots/{dataset}_{data_one_shot['model'][0]}_three_lr{lr}_earlystopper_{early_stopper_it_enabled}_iterstepit{iter_step_it}_iterstepconst{iter_step_const}_iterft{iter_ft}.png")
plt.show()


print("oneshot", filtered_one_shot.to_string())
print("iterative", filtered_iterative.to_string())
print("constant", filtered_constant.to_string())

