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
            # (1/cp.log(2)) * cp.sum([-cp.log(self.p[j]) for j in range(1,self.n)]) == self.M - self.B,
            self.p >= 1/(2**self.J),
            self.p <= 1,
            self.p[0] == 1
        ]
        # Define optimization problem
        problem = cp.Problem(self.objective, self.constraints)
    
        # Solve the problem
        problem.solve(solver = cp.CLARABEL, gp='True')

        # Get optimal values
        optimal_p = self.p.value
        optimal_value = problem.value
        rounded_optimal_lengths = np.round(np.log2(1/optimal_p))
        return optimal_p, optimal_value, rounded_optimal_lengths

if __name__ == "__main__":
    solver = ParityCheckOptimiser(constants.B, constants.n, constants.K, constants.J, constants.M, constants.EPSILON_TREE)
    optimal_p, optimal_value, lengths = solver.solve()

    print("Optimal p:", optimal_p)
    print("Optimal objective value:", optimal_value)
    print("Lengths", lengths)