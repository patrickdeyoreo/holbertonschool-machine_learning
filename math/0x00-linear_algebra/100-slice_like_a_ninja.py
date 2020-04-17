#!/usr/bin/env python3
"""Provides a function to slice a numpy.ndarray along specific axes"""
import numpy as np


def np_slice(matrix, axes={}):
    """Slices a numpy.ndarray along specific axes"""
    return matrix[tuple(slice(*axes.get(level, (0, length)))
                        for level, length in enumerate(matrix.shape))]
