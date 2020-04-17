#!/usr/bin/env python3
"""Provides a function to concatenate two n-dimensional matrices"""


def cat_matrices(mat1, mat2, axis=0):
    """Concatenates two n-dimensional matrices along a specific axis"""
    shape1 = matrix_shape(mat1)
    shape2 = matrix_shape(mat2)
    del shape1[axis]
    del shape2[axis]
    if shape1 == shape2:
        mat1 = matrix_copy(mat1)
        mat2 = matrix_copy(mat2)
    return None


def cat_matrices_map(mat1, mat2, axis=0):
    """Recursive map wrapper to concatenate two n-dimensional matrices"""
    if len(mat1) != len(mat2):
        raise ValueError
    if isinstance(mat1[0], list) and isinstance(mat2[0], list):
        return list(map(add_matrices_map, mat1, mat2))
    return [a + b for a, b in zip(mat1, mat2)]


def matrix_shape(matrix):
    """Computes the shape of a matrix"""
    shape = [len(matrix)]
    if len(matrix) != 0 and isinstance(matrix[0], list):
        shape += matrix_shape(matrix[0])
    return shape


def matrix_copy(matrix):
    """Perform a deep copy of a matrix"""
    if len(matrix) != 0 and isinstance(matrix[0], list):
        return list(map(matrix_copy, matrix))
    return matrix[:]
