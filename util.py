import numpy as np

def decimal_numbers_to_binary_vectors(decimal_numbers, width):
    bit_strings = [np.binary_repr(decimal_number, width = width) for decimal_number in decimal_numbers];
    return np.array([np.array([int(digit) for digit in bit_string]) for bit_string in bit_strings]).T

def binary_vectors_to_decimal_numbers(binary_vectors):
    return binary_vectors.T @ 2 ** np.arange(binary_vectors.shape[0] - 1, -1, -1)