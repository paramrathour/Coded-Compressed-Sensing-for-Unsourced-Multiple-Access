import numpy as np
from scipy.io import loadmat

RANDOM_SEED = 42 # Set this to None for truly (pseudo)random behavior.

B = 101

n = 11
Ka = 25
Kd = 10
K = Ka + Kd

J = 15
# J = 14
M = n*J

EPSILON_TREE = 0.0025
# EPSILON_CS = 1e-3

BCH = (2047,23) # (Length, Field) of the Bose–Chaudhuri–Hocquenghem code
N = BCH[0] * n

# A = np.zeros((N//n, 2**J))
# A = np.random.choice([-1,1], size = (N//n, 2**J))
data = loadmat('BCH_matrix.mat')
A = data['A1']
A = 2*A-1;

parity_lengths = None
G = None
