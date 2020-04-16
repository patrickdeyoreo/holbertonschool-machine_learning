#!/usr/bin/env python3
"""Provides a function to concatenate two 2-dimensional matrices"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenates two 2D matrices"""
    if axis == 0:
        return [u.copy() for u in mat1] + [v.copy() for v in mat2]
    else:
        return [u + v for u, v in zip(mat1, mat2)]
