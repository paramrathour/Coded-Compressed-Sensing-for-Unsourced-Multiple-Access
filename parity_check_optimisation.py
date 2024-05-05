import cvxpy as cp
import numpy as np
import math

import constants

class ParityCheckOptimiser:
    def __init__(self, B, n, K, J, M, epsilon_tree):
        self.B = B
        self.n = n
        self.K = K
        self.J = J
        self.M = M
        self.epsilon_tree = epsilon_tree


    def solve(self):
        # Define variables
        self.p = cp.Variable(self.n, pos = True)
 
        # Define expected value of tilde_L
        expected_tilde_L = [cp.sum([self.K**(j-q) * (self.K - 1) * cp.prod(self.p[q:j+1]) for q in range(1, j+1)]) for j in range(1, self.n)]

        # Define expected value of tilde_C_tree
        expected_tilde_C_tree = (self.n-1) * self.K + cp.sum(expected_tilde_L[0:-1]) * self.K

        # Define objective function
        self.objective = cp.Minimize(expected_tilde_C_tree)

        # Define constraints
        self.constraints = [
            expected_tilde_L[-1] <= self.epsilon_tree,
            1 / cp.prod(self.p[1:]) == cp.exp(cp.log(2) * (self.M - self.B)),
            self.p >= 1/(2**self.J),
            self.p <= 1,
            self.p[0] == 1
        ]
        print(self.constraints[1].is_dgp())
        # Define optimization problem
        problem = cp.Problem(self.objective, self.constraints)
    
        # Solve the problem
        problem.solve(solver = cp.CLARABEL, gp='True') # CLARABEL replaces ECOS solver as the default solver in future versions

        # Get optimal values
        optimal_p = self.p.value
        optimal_objective_value = problem.value
        optimal_lengths = np.round(np.log2(1 / optimal_p))
        return optimal_lengths, optimal_p, optimal_objective_value

if __name__ == "__main__":
    solver = ParityCheckOptimiser(constants.B, constants.n, constants.K, constants.J, constants.M, constants.EPSILON_TREE)
    lengths, p, objective_value = solver.solve()

    print("Optimal lengths:", lengths)
    print("Optimal p:", p)
    print("Optimal objective value:", objective_value)
#######################################################
import cvxpy as cp
import numpy as np
import math

import constants

def solve_parity_check_optimization(B, n, K, J, M, epsilon_tree):
    # Define variables
    p = cp.Variable(n, pos=True)
 
    # Define expected value of tilde_L
    expected_tilde_L = [cp.sum([K**(j-q) * (K - 1) * cp.prod(p[q:j+1]) for q in range(1, j+1)]) for j in range(1, n)]

    # Define expected value of tilde_C_tree
    expected_tilde_C_tree = (n-1) * K + cp.sum(expected_tilde_L[0:-1]) * K

    # Define objective function
    objective = cp.Minimize(expected_tilde_C_tree)

    # Define constraints
    constraints = [
        expected_tilde_L[-1] <= epsilon_tree,
        1 / cp.prod(p[1:]) == cp.exp(cp.log(2) * (M - B)),
        p >= 1/(2**J),
        p <= 1,
        p[0] == 1
    ]
    print(constraints[1].is_dgp())
    # Define optimization problem
    problem = cp.Problem(objective, constraints)
    
    # Solve the problem
    problem.solve(solver=cp.CLARABEL, gp=True)  # CLARABEL replaces ECOS solver as the default solver in future versions

    # Get optimal values
    optimal_p = p.value
    optimal_objective_value = problem.value
    optimal_lengths = np.round(np.log2(1 / optimal_p))
    
    return optimal_lengths, optimal_p, optimal_objective_value

if __name__ == "__main__":
    B = constants.B
    n = constants.n
    K = constants.K
    J = constants.J
    M = constants.M
    epsilon_tree = constants.EPSILON_TREE
    
    lengths, p, objective_value = solve_parity_check_optimization(B, n, K, J, M, epsilon_tree)

    print("Optimal lengths:", lengths)
    print("Optimal p:", p)
    print("Optimal objective value:", objective_value)
