#!/usr/bin/env python3
"""Provides a function to perform addition of two n-dimensional matrices"""


def add_matrices(mat1, mat2):
    """Performs element-wise addition of two n-dimensional matrices"""
    if isinstance(mat1[0], list) and isinstance(mat2[0], list):
        return list(map(add_matrices, mat1, mat2))
    return [a + b for a, b in zip(mat1, mat2)]
