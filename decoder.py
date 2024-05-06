import numpy as np
import scipy.linalg
import pickle

import constants
from util import decimal_numbers_to_binary_vectors
from encoder import run_encoding

def decoding_cs(Y):
    """
    Inputs
    Y - N/n×n matrix (2D-numpy array)
    A - N/n×2^J matrix (2^J codewords of BCH code)
    Outputs
    """
    L = np.zeros((constants.n, constants.J, constants.K))
    for i in range(constants.n):
        print("Sub-block:", i)
        x = omp(Y[:, i], constants.A, constants.K)
        indices = np.where(x == 1)[0]
        L[i] = decimal_numbers_to_binary_vectors(indices, constants.J)
    return L

def decoding_tree(L):
    W = [L[i, :constants.J-constants.parity_lengths[i], :] for i in range(constants.n)]
    P = [L[i, constants.J-constants.parity_lengths[i]:, :] for i in range(constants.n)]

    path = [0]
    result = []
    result_partial = []
    for i in range(constants.K):
        temp = []
        decoding_tree_backtracking(W, P, [i], result, temp)
        if len(temp) != 0:
            result_partial.append(max(temp, key = len))

    messages = np.zeros((constants.B, len(result)))
    messages_partial = np.zeros((constants.B, len(result_partial)))
    if len(result) == 0:
        print("Can't recover any message fully, trying partial recovery...")
    else:
        print(result)
        for i, r in enumerate(result):
            messages[:, i] = np.concatenate([W[j][:, r[j]] for j in range(constants.n)])

    if len(result_partial) == 0:
        print("Can't recover any fragment of any message fully")
        return
    
    print("Found partial messages...")
    print(result_partial)
    for i, r in enumerate(result_partial):
        # print(np.concatenate([W[j][:, s] for j,s in enumerate(r)]))
        m = np.concatenate([W[j][:, s] for j,s in enumerate(r)])
        messages_partial[:len(m), i] = m
    # print(messages_partial)

    print("Each message size", constants.B)
    # print(messages_sent.shape[1], "messages sent", messages_sent)
    print(messages.shape[1], "messages received completely", messages)
    print("At least", messages_partial.shape[1]-constants.Kd, "messages received partialy", messages_partial)

    return messages, messages_partial

def decoding_tree_backtracking(W, P, path, result, temp):
    depth = len(path)
    if depth == constants.n:
        result.append(path[:])
        return True

    for i in range(constants.K):
        path.append(i)
        # print("Checking path: ", path)
        sub_blocks = [W[j][:, p] for j,p in enumerate(path)]
        parity_bits = sum(constants.G[(j, depth-1)] @ sub_blocks[j] for j in range(depth)) % 2
        if np.array_equal(parity_bits, P[depth][:, i]):
            temp.append(np.copy(path))
            if decoding_tree_backtracking(W, P, path, result, temp) == True:
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
    L = decoding_cs(Y)
    with open("data-L.pkl", 'wb') as file:
        pickle.dump(L, file)
    # with open("data-L.pkl", 'rb') as file:
    #     L = pickle.load(file)
    messages_received_complete, messages_received_partial  = decoding_tree(L)
    print("Each message size", constants.B)
    print(messages_sent.shape[1], "messages sent", messages_sent)
    with open("messages.txt", "w", encoding="ascii") as file:
        np.set_printoptions(threshold=np.inf)
        file.write("Sent messages\n")
        file.write(repr(messages_sent))
        file.write("\n\n")
        file.write("Completely received messages\n")
        file.write(repr(messages_received_complete))
        file.write("\n\n")
        file.write("Partialy received messages\n")
        file.write(repr(messages_received_partial))

if __name__ == "__main__":
    run_decoding()