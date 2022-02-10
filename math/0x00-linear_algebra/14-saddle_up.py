#!/usr/bin/env python3
"""
Provide a function to perform matrix multiplication of two numpy arrays.
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Perform matrix multiplication of two numpy arrays.
    """
    return mat1 @ mat2

