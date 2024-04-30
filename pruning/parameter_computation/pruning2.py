import numpy as np


def find_number_of_steps(pruning_rate, target_pruned_rate):
    """
    Calculate the number of steps required to achieve or exceed a target total pruning rate
    given a specific pruning rate per step in iterative geometric pruning. This function also
    prints the fraction of original weights pruned at each step.

    Parameters:
    pruning_rate (float): The pruning rate per step.
    target_pruned_rate (float): The total target pruning rate to be achieved.

    Returns:
    int: The number of steps required to achieve or exceed the target pruning rate.
    """
    if not (0 < pruning_rate < 1):
        raise ValueError("Pruning rate must be between 0 and 1 (exclusive).")
    if not (0 < target_pruned_rate < 1):
        raise ValueError("Target pruning rate must be between 0 and 1 (exclusive).")

    total_pruned = 0
    steps = 0
    while total_pruned < target_pruned_rate:
        steps += 1
        fraction_pruned_this_step = (1 - pruning_rate) ** (steps - 1) * pruning_rate
        total_pruned += fraction_pruned_this_step
        print(f"Step {steps}: Fraction of original weights pruned this step = {fraction_pruned_this_step:.4f}")

    print(f"Total fraction pruned after {steps} steps: {total_pruned:.4f}")
    return steps, total_pruned

if __name__=="__main__":
    # Example usage:
    pruning_rate = 0.05  # Pruning rate per step
    target_pruned_rate = 0.96  # Target total pruned rate
    steps_required = find_number_of_steps(pruning_rate, target_pruned_rate)
    print(f"Number of steps required to achieve at least {target_pruned_rate * 100:.2f}% pruning: {steps_required}")