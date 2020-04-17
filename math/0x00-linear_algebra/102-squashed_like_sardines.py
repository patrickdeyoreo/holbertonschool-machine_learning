#!/usr/bin/env python3
"""Provides a function to concatenate two n-dimensional matrices"""


def cat_matrices(mat1, mat2, axis=0):
    """Concatenates two n-dimensional matrices along a specific axis"""
    try:
        return cat_matrices_map(mat1, mat2)
    except ValueError:
        return None


def cat_matrices_map(mat1, mat2, axis=0):
    """Recursive map wrapper to concatenate two n-dimensional matrices"""
    if len(mat1) != len(mat2):
        raise ValueError
    if isinstance(mat1[0], list) and isinstance(mat2[0], list):
        return list(map(add_matrices_map, mat1, mat2))
    return [a + b for a, b in zip(mat1, mat2)]
