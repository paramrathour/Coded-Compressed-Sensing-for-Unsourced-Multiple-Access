import cvxpy as cp
import numpy as np
import math

import constants

def parityCheckOptimiser():
    # Define variables
    p = cp.Variable(constants.n, pos=True)
 
    # Define expected value of tilde_L
    expected_tilde_L = [cp.sum([constants.Ka**(j-q) * (constants.Ka - 1) * cp.prod(p[q:j+1]) for q in range(1, j+1)]) for j in range(1, constants.n)]

    # Define expected value of tilde_C_tree
    expected_tilde_C_tree = (constants.n-1) * constants.Ka + cp.sum(expected_tilde_L[:-1]) * constants.Ka

    # Define objective function
    objective = cp.Minimize(expected_tilde_C_tree)

    # Define constraints
    constraints = [
        expected_tilde_L[-1] <= constants.EPSILON_TREE,
        1 / cp.prod(p[1:]) == cp.exp(cp.log(2) * (constants.M - constants.B)),
        p >= 1/(2**constants.J),
        p <= 1,
        p[0] == 1
    ]
    
    # Define optimization problem
    problem = cp.Problem(objective, constraints)
    
    # Solve the problem
    problem.solve(solver=cp.CLARABEL, gp=True)  # CLARABEL replaces ECOS solver as the default solver in future versions

    # Get optimal values
    optimal_p = p.value
    optimal_objective_value = problem.value
    optimal_lengths = (np.round(np.log2(1 / optimal_p))).astype(int)
    
    # assert(sum(optimal_lengths) == constants.M - constants.B)
    diff = sum(optimal_lengths) - (constants.M - constants.B)
    assert(abs(diff) < constants.n)
    if diff > 0:
        optimal_lengths[1:diff+1] -=  1
    if diff < 0:
        optimal_lengths[diff:] +=  1
    return optimal_lengths, optimal_p, optimal_objective_value

def generateParityMatrices():
    G = {}
    for j in range(constants.n-1):
        for i in range(j+1):
            G[(i,j)] = np.random.randint(2, size = (constants.parity_lengths[j+1], constants.J-constants.parity_lengths[i]))
    return G

def generateParityBits(W):
    """
    W = [w_0, w_1, ..., w_(n-1)], w_i is ith sub-block as a vector
    """
    P = [None] * (constants.n-1)
    for i in range(constants.n-1):
        P[i] = sum(constants.G[(j, i)] @ W[j] for j in range(i+1)) % 2
    return P

def run_parity():
    np.random.seed(constants.RANDOM_SEED)

    constants.parity_lengths, p, objective_value = parityCheckOptimiser()
    print("Optimal lengths:", constants.parity_lengths)
    # print("Optimal p:", p)
    print("Optimal objective value:", objective_value)

    constants.G = generateParityMatrices()
    # for i,J in constants.G.items():
    #     print(i, J.shape)

if __name__ == "__main__":
    run_parity()