#!/usr/bin/env python3
"""Provides a function to convert a numeric-label vector to a one-hot matrix"""
# pylint: disable=invalid-name

import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a numeric-label vector to a one-hot matrix
    Arguments:
        one_hot: one-hot encoded numpy.ndarray with shape (classes, m), where
                 classes is the maximum number of classes, and
                 m is the number of examples
        classes: the maximum number of classes found in Y
    Return:
        numpy.ndarray with shape (m,) containing the numeric labels for each
        example, or None on failure
    """
    try:
        if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
            return None
        return np.argmax(one_hot, axis=0)
        # return np.array([row.tolist().index(1) for row in one_hot.T])
    except (TypeError, ValueError):
        return None
