#!/usr/bin/env python3
"""Provides a function to concatenate two n-dimensional matrices"""


def cat_matrices(mat1, mat2, axis=0):
    """Concatenates two n-dimensional matrices along a specific axis"""
    shape1 = matrix_shape(mat1)
    shape2 = matrix_shape(mat2)
    del shape1[axis]
    del shape2[axis]
    return None if shape1 != shape2 else cat_matrices_comp(mat1, mat2, axis)


def cat_matrices_comp(mat1, mat2, axis):
    """Recursive comprehension to concatenate two n-dimensional matrices"""
    if axis != 0:
        return [cat_matrices_comp(u, v, axis - 1) for u, v in zip(mat1, mat2)]
    return matrix_copy(mat1 + mat2)


def matrix_copy(matrix):
    """Performs a deep copy of a matrix"""
    if len(matrix) != 0 and isinstance(matrix[0], list):
        return list(map(matrix_copy, matrix))
    return matrix[:]


def matrix_shape(matrix):
    """Computes the shape of a matrix"""
    shape = [len(matrix)]
    if len(matrix) != 0 and isinstance(matrix[0], list):
        shape += matrix_shape(matrix[0])
    return shape
