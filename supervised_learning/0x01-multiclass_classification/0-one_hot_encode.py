#!/usr/bin/env python3
"""Provides a function to convert a numeric-label vector to a one-hot matrix"""
# pylint: disable=invalid-name

import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric-label vector to a one-hot matrix
    Arguments:
        Y: numpy.ndarray with shape (m,) containing numeric class labels, where
           m is the number of examples
        classes: the maximum number of classes found in Y
    Return:
        one-hot encoding of Y with shape (classes, m), or None on failure
    """
    try:
        if not isinstance(Y, np.ndarray) or classes <= Y.max():
            return None
        return np.array([np.where(Y == n, 1.0, 0.0) for n in range(classes)])
        # return np.array([[float(y == n) for y in Y] for n in range(classes)])
    except (TypeError, ValueError):
        return None
