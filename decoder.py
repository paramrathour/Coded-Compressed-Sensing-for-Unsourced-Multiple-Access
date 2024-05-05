import numpy as np
import scipy.linalg
import pickle

import constants
from util import decimal_numbers_to_binary_vectors
from encoder import run_encoding

def decoding_cs(Y, A):
    """
    Inputs
    Y - N/n×n matrix (2D-numpy array)
    A - N/n×2^J matrix (2^J codewords of BCH code)
    Outputs
    """
    L = np.zeros((constants.n, constants.J, constants.K))
    for i in range(constants.n):
        print("Sub-block:", i)
        x = omp(Y[:, i], A, constants.K)
        indices = np.where(x == 1)[0]
        L[i] = decimal_numbers_to_binary_vectors(indices, constants.J)
    return L

def decoding_tree(L):
    W = [L[i, :constants.J-constants.parity_lengths[i], :] for i in range(constants.n)]
    P = [L[i, constants.J-constants.parity_lengths[i]:, :] for i in range(constants.n)]
    for i in W:
        print(i.shape)
    print("and")
    for i in P:
        print(i.shape)
    path = [0]
    result = []

    for i in range(constants.K):
        decoding_tree_backtracking(W, P, [i], result)

    if len(result) == 0:
        print("Failure! Can't recover any messages")
        return

    messages = np.zeros((J, len(result)))
    for i, r in enumerate(result):
        messages[:, i] = [W[j][r[j]] for j in range(constants.n)]
    return messages

def decoding_tree_backtracking(W, P, path, result):
    depth = len(path)
    if depth == constants.n-1:
        result.append(path[:])
        return True

    for i in range(constants.K):
        path.append(i)
        print("Checking path: ", path)
        sub_blocks = [W[j][:, p] for j,p in enumerate(path)]
        parity_bits = sum(constants.G[(j, depth-1)] @ sub_blocks[j] for j in range(depth)) % 2
        if np.array_equal(parity_bits, P[depth][:, i]):
            if decoding_tree_backtracking(W, P, path, result) == True:
                return True
        path.pop()

    return False

def omp(y, A, sparsity):
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

def run_decoding():
    messages_sent, Y = run_encoding()
    L = decoding_cs(Y, constants.A)
    # with open("data-L.pkl", 'wb') as file:
    #     pickle.dump(L, file)
    # with open("data-L.pkl", 'rb') as file:
    #             L = pickle.load(file)
    messages_received  = decoding_tree(L)

    print(messages_sent)
    print(messages_received)
if __name__ == "__main__":
    run_decoding()