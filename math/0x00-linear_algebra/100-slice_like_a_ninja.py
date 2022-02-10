#!/usr/bin/env python3
"""
Provide a function to slice a numpy array along specific axes.
"""


def np_slice(matrix, axes={}):
    """
    Slice a numpy array along specific axes.
    """
    slices = (slice(*axes.get(depth, (None, None)))
              for depth in range(len(matrix.shape)))
    return matrix[tuple(slices)]
