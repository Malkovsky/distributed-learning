import numpy as np
import cvxpy as cp

def find_optimal_weights(graph):
    '''
    graph: list of pairs describing edges, e.g. [(0, 1), (0, 2), (1, 3)]
    Returns a list of corresponding weights and a convergence factor (lambda_2 of (I - L))
    '''
    verticies = dict()
    for (u, v) in graph:
        if u not in verticies:
            verticies[u] = len(verticies)
        if v not in verticies:
            verticies[v] = len(verticies)
    n = len(verticies)
    gamma = cp.Variable()
    weights = cp.Variable(len(graph))
    A = np.zeros((n, len(graph)))
    for i, (u, v) in enumerate(graph):
        if u != v:
            A[verticies[u], i] = 1
            A[verticies[v], i] = -1
    L = A @ cp.diag(weights) @ A.T
    ones_n = np.ones((n, n)) / n
    constraints = [
                   (np.eye(n) - L - ones_n) >> (-gamma * np.eye(n)),
                   (np.eye(n) - L - ones_n) << ( gamma * np.eye(n)),
                    L >> 0,
                  ]
    problem = cp.Problem(cp.Minimize(gamma), constraints)
    problem.solve()
    return weights.value, gamma.value.item()
    