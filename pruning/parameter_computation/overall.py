import numpy as np
from sklearn.model_selection import ParameterGrid

from pruning2 import find_number_of_steps
from pruning import calculate_iter_rate

pruning_rate = [0.99, 0.96, 0.92, 0.88, 0.8, 0.7]
iter_rate =[0.02, 0.05, 0.1]

params=ParameterGrid({"pruning_rate": pruning_rate, "iter_rate": iter_rate})


triples = []

for p in params:
    ir = p["iter_rate"]
    pr = p["pruning_rate"]
    steps, total_pruned = find_number_of_steps(ir,pr)
    if np.abs(pr-total_pruned)>0.0001:
        print(pr, steps)
        ir, total_pruned = calculate_iter_rate(steps, pr)
    triples.append([pr, ir, steps, total_pruned])

print(triples)