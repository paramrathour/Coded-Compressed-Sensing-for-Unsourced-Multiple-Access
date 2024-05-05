# Encoder file
#  Step 1 Tree encoding 
#  Step 2 CS Encoding

import constants
import numpy as np
from parity import run_parity, parityCheckOptimiser, generateParityBits, generateParityMatrices
from util import binary_vectors_to_decimal_numbers

""" 
By importing constants,I would be having
B,n,Ka,Kd,K,J,M,EPSILON_TREE
accessible by constants.B
"""
def generateMessages():
    messages_binary_vector = np.random.randint(2, size = (constants.B, constants.Ka))
    messages_decimal_number = binary_vectors_to_decimal_numbers(messages_binary_vector)
    # messages_decimal_number = np.random.randint(0, 2**constants.B, size = constants.Ka)
    # messages_binary_vector = decimal_numbers_to_binary_vectors(messages_decimal_number, constants.B)
    code_lengths = constants.J - constants.parity_lengths
    print("Code lengths: ", code_lengths)
    W = np.split(messages_binary_vector, np.cumsum(code_lengths[:-1]))
    return messages_binary_vector, W

def encoding_tree(W):
    """
    Input
    W = [[W_0], [W_1], ..., [W_(n-1)]], W_i is a matrix containing all the ith sub-blocks as its columns
    Output
    V = [[V_0], [V_1], ..., [V_(n-1)]], parity bits added
    """
    P = [None] * constants.Ka
    for i in range(constants.Ka):
        P[i] = generateParityBits([W[j][:,i] for j in range(constants.n-1)])
    P = [np.vstack([row[i] for row in P]).T for i in range(len(P[0]))];
    V = [W[0]] + [np.concatenate([W[i+1], P[i]]) for i in range(constants.n-1)]
    return V
    # V = np.vstack([W[0], np.vstack(temp)]) # original problem without sub-blocks is one matrix instead of an array of n matrices

def encoding_cs(V):
    y = np.zeros((constants.N//constants.n, constants.n))
    for i in range(constants.n):
        y[:, i] = np.sum(constants.A[:, binary_vectors_to_decimal_numbers(V[i])])
    return y

def run_encoding():
    run_parity()
    messages_sent, W = generateMessages()
    # for i in W:
    #     print(i.shape)
    V = encoding_tree(W)
    # for i in V:
    #     print(i.shape)
    Y = encoding_cs(V)
    # print(y.shape)
    return messages_sent, Y

if __name__ == "__main__":
    run_encoding()

# def w_to_v(w):
#     information_bits_length = constants.J-parity_bits_length
#     # converting input 'w' into sub-blocks of length specified by information_Bits_length
#     w_in_subblock_form = []
#     start = 0
#     for length in information_bits_length:
#         end=start+length 
#         w_in_subblock_form.append(w[start:end])
#         start=end
    
#     # Now construct the parity bits assuming the constants file has 
#     # random matrices defined which can be accessed using G[(i,j)] i rows and j columns
#     # paritybits corresponding to j_th subblock corresponds to {p}(j)=\sum_{\ell=0}^{j-1}G_{\ell,j-1} {w\_insubblock\_form}(\ell)
#     parity_bits = []
#     for j in range(1,constants.n):
#         p_j = np.zeros(parity_bits_length[j])
#         for l in range(j):
#             G_j_1_l = constants.G[(j-1,l)]
#             # check dimenions for compatibility of vector matrix product in the next line
            
#             p_j += np.dot(G_j_1_l,w_in_subblock_form[l])
#             # check dimensions of p_j,it should be a column
#             p_j %= 2
#         parity_bits.append(p_j)
#     # combining information and parity bits to form vector v
#     v = []
#     for i in range(constants.n):
#         if i==0:
#             v.append(w_in_subblock_form[i])
#         else:
#             v.append(np.concatenate((w_in_subblock_form[i],parity_bits[i-1])))
#     # v constructed as required
#     # Tree encoding step 1 completed