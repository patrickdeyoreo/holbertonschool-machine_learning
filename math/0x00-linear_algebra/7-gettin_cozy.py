#!/usr/bin/env python3
"""Provides a function to concatenate two 2-dimensional matrices"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenates two 2D matrices along the specified axis"""
    if axis == 0:
        if len(mat1[0]) == len(mat2[0]):
            return [row.copy() for row in mat1 + mat2]
        return None
    else:
        if len(mat1) == len(mat2):
            return [u + v for u, v in zip(mat1, mat2)]
        return None
