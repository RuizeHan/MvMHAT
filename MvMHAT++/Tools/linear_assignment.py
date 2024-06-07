from scipy.optimize import linear_sum_assignment as linear_assignment
import numpy as np


def sklearn_linear_assignment(cost_matrix):
    indices = linear_assignment(cost_matrix)
    indices = np.asarray(indices)
    indices = np.transpose(indices)
    return indices
