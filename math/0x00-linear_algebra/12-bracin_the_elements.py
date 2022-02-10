#!/usr/bin/env python3
"""
Provide a function that adds, subtracts, multiplies and divides numpy arrays.
"""


def np_elementwise(mat1, mat2):
    """
    Add, subtract, multiply and divide numpy arrays.
    """
    return mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2
