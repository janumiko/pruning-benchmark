import numpy as np


def calculate_iter_rate(n, remaining):
    pruned_rates = []
    r = 1-remaining
    """
    Calculate the pruning rate per step for iterative geometric pruning.

    Parameters:
    n (int): The number of steps (iterations).
    r (float): The target remaining fraction of the original weights.

    Returns:
    float: The pruning rate per step.
    """
    if n <= 0:
        raise ValueError("Number of steps (n) must be greater than 0.")
    if not (0 < r < 1):
        raise ValueError("Target remaining fraction (r) must be between 0 and 1 (exclusive).")

    # Calculate the pruning rate per step
    pruning_rate = 1 - r**(1/n)

    p =pruning_rate

    """
    Print the fraction of the original weights pruned at each step for given pruning rate and number of steps.


    Parameters:
    p (float): Pruning rate per step.
    n (int): Total number of steps.
    """
    for k in range(1, n+1):
        fraction_pruned = (1 - p) ** (k - 1) * p
        pruned_rates.append(fraction_pruned)
        print(f"Fraction of original weights pruned at step {k}: {fraction_pruned:.4f}")
    pruned_rates_sum = np.sum(pruned_rates)
    return np.round(pruning_rate,5), np.round(pruned_rates_sum,6)

if __name__=="__main__":
    # Example usage:
    n = 21  # Number of pruning steps
    r = 0.88 # pruned rate
    pruning_rate, pruned_rates_sum = calculate_iter_rate(n, r)
    print(f"Pruning rate per step: {pruning_rate:.4f}")
    print("Total pruned: ", pruned_rates_sum)
