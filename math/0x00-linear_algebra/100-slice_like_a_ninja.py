#!/usr/bin/env python3
"""Provides a function to slice a numpy.ndarray along specific axes"""


def np_slice(matrix, axes={}):
    """Slices a numpy.ndarray along specific axes"""
    part = (slice(*axes.get(depth, (None, None, None)))
            for depth, length in enumerate(matrix.shape))
    return matrix[tuple(part)]
