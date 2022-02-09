#!/usr/bin/env python3
"""Provides a function to perform element-wise additon of two 2D matrices"""


def add_matrices2D(mat1, mat2):
    """Performs element-wise additon of two 2-dimensional matrices"""
    if len(mat1) == len(mat2) and len(mat1[0]) == len(mat2[0]):
        return [[a + b for a, b in zip(u, v)] for u, v in zip(mat1, mat2)]
    return None
