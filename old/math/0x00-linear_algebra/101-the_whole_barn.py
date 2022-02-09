#!/usr/bin/env python3
"""Provides a function to perform addition of two n-dimensional matrices"""


def add_matrices(mat1, mat2):
    """Performs element-wise addition of two n-dimensional matrices"""
    try:
        return add_matrices_map(mat1, mat2)
    except ValueError:
        return None


def add_matrices_map(mat1, mat2):
    """Recursive map wrapper to add two n-dimensional matrices"""
    if len(mat1) != len(mat2):
        raise ValueError
    if isinstance(mat1[0], list) and isinstance(mat2[0], list):
        return list(map(add_matrices_map, mat1, mat2))
    return [a + b for a, b in zip(mat1, mat2)]
