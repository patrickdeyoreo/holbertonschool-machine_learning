#!/usr/bin/env python3
"""
Provide a function that adds two matrices.
"""


import re


class Matrix:
    """
    Define a matrix class to convert matrices to numpy arrays.
    """
    import numpy as np

    def __init__(self, matrix):
        """
        Convert a matrix to a numpy array.
        """
        self.matrix = self.np.array(matrix)


def matrix_shape(matrix):
    """
    Calculate the shape of a matrix.
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
    return shape


def add_matrices(mat1, mat2):
    """
    Add two numpy arrays.
    shape1 = matrix_shape(mat1)
    shape2 = matrix_shape(mat2)
    if shape1 != shape2:
        return None
    shape = shape1
    """
    mat1 = Matrix(mat1)
    mat2 = Matrix(mat2)
    if mat1.matrix.shape == mat2.matrix.shape:
        return mat1.matrix + mat2.matrix
    return None
