#!/usr/bin/env python3
"""Provides a function to slice a numpy.ndarray along specific axes"""
import numpy as np


def np_slice(matrix, axes={}):
    """Slices a numpy.ndarray along specific axes"""
    return matrix[tuple(slice(*axes.get(depth, (0, length)))
                        for depth, length in enumerate(matrix.shape))]
