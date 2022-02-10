#!/usr/bin/python3
"""
Provide a function to concatenate two numpy arrays along a specific axis.
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenate two numpy arrays along a specific axis.
    """
    return np.concatenate((mat1, mat2), axis=axis)
