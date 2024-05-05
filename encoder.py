# Encoder file
#  Step 1 Tree encoding 
#  Step 2 CS Encoding
import constants
import numpy as np
from parity_check_optimisation import solve_parity_check_optimization
""" 
By importing constants,I would be having
B,n,Ka,Kd,K,J,M,EPSILON_TREE
accessible by constants.B
"""

parity_bits_length, p, objective_value = solve_parity_check_optimization(B, n, K, J, M, epsilon_tree)


def w_to_v(w):
    information_bits_length = constants.J-parity_bits_length
    # converting input 'w' into sub-blocks of length specified by information_Bits_length
    w_in_subblock_form = []
    start = 0
    for length in information_bits_length:
        end=start+length 
        w_in_subblock_form.append(w[start:end])
        start=end
    
    # Now construct the parity bits assuming the constants file has 
    # random matrices defined which can be accessed using G[(i,j)] i rows and j columns
    # paritybits corresponding to j_th subblock corresponds to {p}(j)=\sum_{\ell=0}^{j-1}G_{\ell,j-1} {w\_insubblock\_form}(\ell)
    parity_bits = []
    for j in range(1,constants.n):
        p_j = np.zeros(parity_bits_length[j])
        for l in range(j):
            G_j_1_l = constants.G[(j-1,l)]
            # check dimenions for compatibility of vector matrix product in the next line
            
            p_j += np.dot(G_j_1_l,w_in_subblock_form[l])
            # check dimensions of p_j,it should be a column
            p_j %= 2
        parity_bits.append(p_j)
    # combining information and parity bits to form vector v
    v = []
    for i in range(constants.n):
        if i==0:
            v.append(w_in_subblock_form[i])
        else:
            v.append(np.concatenate((w_in_subblock_form[i],parity_bits[i-1])))
    # v constructed as required
    # Tree encoding step 1 completed
    
    
    