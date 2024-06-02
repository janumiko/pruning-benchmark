import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
import heapq



def find_newest_folders(directory, prefix, num_folders=3):
    # List all the folders in the given directory that match the prefix
    folders = [f for f in os.listdir(directory) if f.startswith(prefix)]

    # Use a min-heap to keep the top num_folders folders based on their date and time
    top_folders_heap = []

    # Loop through each folder
    for folder in folders:
        # Extract the date and time part of the folder name
        try:
            parts = folder.split('_', 2)
            if len(parts) < 3:
                continue  # Skip folders that do not have the correct format
            date_time_str = parts[1] + '_' + parts[2]
            folder_datetime = datetime.datetime.strptime(date_time_str, '%Y-%m-%d_%H-%M-%S')

            # Use a tuple (folder_datetime, folder) to keep track in the heap
            if len(top_folders_heap) < num_folders:
                heapq.heappush(top_folders_heap, (folder_datetime, folder))
            else:
                # Only push to the heap if the current folder is newer than the smallest one in the heap
                heapq.heappushpop(top_folders_heap, (folder_datetime, folder))

        except (IndexError, ValueError):
            # Handle cases where the folder name format is incorrect
            continue

    # Extract the folders from the heap and sort them to get the most recent first
    top_folders_heap.sort(reverse=True, key=lambda x: x[0])


    return [folder.split('_', 1)[1] for _, folder in top_folders_heap]

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
d2 = find_newest_folders("../csvs", dataset)
# date="_2024-05-15_11-25-32"
# dataset="cifar100"
#date="_"+d

#data = pd.read_csv(f"../csvs/{dataset}{date}/pruning_results_{dataset}_manual{date}.csv")

data1 = pd.read_csv(f"../csvs/{dataset}_{d2[0]}/pruning_results_{dataset}_manual_{d2[0]}.csv")
data2 = pd.read_csv(f"../csvs/{dataset}_{d2[1]}/pruning_results_{dataset}_manual_{d2[1]}.csv")
data3 = pd.read_csv(f"../csvs/{dataset}_{d2[2]}/pruning_results_{dataset}_manual_{d2[2]}.csv")

# files
data1 = pd.read_csv("/home/kamil/Dropbox/Current_research/compression/pruning-benchmark/csvs/cifar100_2024-05-31_09-37-02/pruning_results_cifar100_manual_2024-05-31_09-37-02.csv")
data2 = pd.read_csv("../csvs/cifar100_2024-05-31_09-45-18/pruning_results_cifar100_manual_2024-05-31_09-45-18.csv")
data3 = pd.read_csv("../csvs/cifar100_2024-05-31_09-47-22/pruning_results_cifar100_manual_2024-05-31_09-47-22.csv")

data = pd.concat([data1, data2, data3], axis=0)


# Create separate dataframes for each 'type'
data_one_shot = data[data['type'] == 'manual_one_shot']
data_constant = data[data['type'] == 'manual_constant']
data_iterative = data[data['type'] == 'manual_geometric']

# Load the datasets
#data_one_shot = pd.read_csv(f"../csvs/{dataset}{date}/pruning_results_{dataset}_one-shot{date}.csv")
#data_iterative = pd.read_csv(f"../csvs/{dataset}{date}/pruning_results_{dataset}_iterative{date}.csv")
#data_constant = pd.read_csv(f"../csvs/{dataset}{date}/pruning_results_{dataset}_constant{date}.csv")

# Function to filter and process data based on learning rate and pruning percentages
# Function to filter and process data based on learning rate, pruning percentages, and specific conditions for iterative strategy
# Function to filter and process data based on learning rate, pruning percentages, and additional conditions
def filter_data(data, pruning_percentages, learning_rate, finetune_epochs=None, scheduler_step=None, early_stopper_patience=None, early_stopper_it_enabled=False, name=None, norm=None):

    print("Norm: " , norm)
    if name=='oneshot':
        pruning_percentages=pruning_percentages[0]
    elif name == 'constant':
        pruning_percentages=pruning_percentages[1]
    elif name == 'iterative':
        pruning_percentages=pruning_percentages[2]

    condition = (
        (data['optimizer.learning_rate'] == learning_rate) &
        (data['pruned_precent'].isin(pruning_percentages))
    )
    # if norm is not None:
    #     condition &= (data['pruning.method.norm'] == norm)
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
# pruning_percentages = [[55.6, 65.33, 75.11, 84.84],[60.15, 72.12, 84.84, 92.28], [60.15, 72.38, 84.84, 92.28]]
pruning_percentages = [[55.6, 65.33, 75.11, 84.84],[55.91, 64.75, 74.97, 84.84], [55.6, 65.33, 75.11, 84.84]]

norm=2
# Filter data ------------------------------------------
lr=0.001; iter_step_it=None; iter_step_const=None; iter_ft=None; iter_early_stopper_patience=None; early_stopper_it_enabled=True
filtered_one_shot = filter_data(data_one_shot, pruning_percentages, 0.001, name="oneshot", norm=norm)
filtered_iterative = filter_data(data_iterative, pruning_percentages, learning_rate=lr, name="iterative", scheduler_step=iter_step_it, finetune_epochs=iter_ft, early_stopper_patience=iter_early_stopper_patience, early_stopper_it_enabled=early_stopper_it_enabled, norm=norm)
filtered_constant = filter_data(data_constant, pruning_percentages, 0.001, name="constant", finetune_epochs=iter_ft, scheduler_step=iter_step_const, early_stopper_patience=iter_early_stopper_patience, early_stopper_it_enabled=early_stopper_it_enabled, norm=norm)

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
plt.title(f'{dataset}: Comparison of Top-1 Accuracies by Pruning Strategy for \n lr {lr}, norm {norm}, early stopper {early_stopper_it_enabled}, iter_step iter {iter_step_it}, iter_step const {iter_step_const}, iter_early_stopper_patience {iter_early_stopper_patience}, iter_finetune {iter_ft}')
plt.legend()
plt.grid(True)
plt.savefig(f"plots/{dataset}_structured_three_lr{lr}_earlystopper_{early_stopper_it_enabled}_iterstepit{iter_step_it}_iterstepconst{iter_step_const}_iterft{iter_ft}_norm{norm}.png")
plt.show()


print("oneshot", filtered_one_shot.to_string())
print("iterative", filtered_iterative.to_string())
print("constant", filtered_constant.to_string())

