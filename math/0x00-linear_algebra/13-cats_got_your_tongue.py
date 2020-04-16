#!/usr/bin/env python3
"""Provides a function to concatenate two numpy.ndarray's"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concatenates two numpy.ndarray's along the specified axis"""
    return np.concatenate((mat1, mat2), axis=axis)
