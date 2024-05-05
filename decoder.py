import numpy as np
import scipy.linalg

import math

import constants

def decoder(Y, A, K, parity_lengths)
    L = decoding_cs(Y, A)

def decoding_cs(Y, A)
    """
    Inputs
    Y - N/n×n matrix (2D-numpy array)
    A - N/n×2^J matrix (2^J codewords of BCH code)
    Outputs
    """
    L = np.zeros((constants.n, constants.J, constants.K))
    for i in range(constants.n):
        x = omp(A, y, constants.K)
        indices = np.where(x == 1)[0]
        information_bits_all = [np.binary_repr(index, width = constants.J) for index in indices];
        L[i] = np.array([np.array([int(digit) for digit in information_bits]) for information_bits in information_bits_all]).T
    return L

def omp(y, A, sparsity)
    """
    A modified version of OMP which returns a binary vector of appropriate sparsity
    Inputs
    A - m×n matrix (2D-numpy array)
    y - m×1 vector (1D-numpy array)
    Output
    x - n×1 vector (1D-numpy array)
    """
    r = np.copy(y)
    T = np.array([], dtype = int)
    for i in range(sparsity):
        j = np.argmax(r.T @ A)
        T = np.union1d(T, j)
        theta = scipy.linalg.pinv(A[:, T]) @ y
        r = y - A[:, T] @ theta

    x = np.zeros(A.shape[1], dtype = np.uint8)
    x[T] = 1
    return x

if __name__ == "__main__":
    solver = ParityCheckOptimiser(constants.B, constants.n, constants.K, constants.J, constants.M, constants.EPSILON_TREE)
    lengths, p, objective_value = solver.solve()

    print("Optimal lengths:", lengths)
    print("Optimal p:", p)
    print("Optimal objective value:", objective_value)
