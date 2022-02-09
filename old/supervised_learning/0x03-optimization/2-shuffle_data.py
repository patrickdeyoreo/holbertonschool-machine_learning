#!/usr/bin/env python3
"""Provides a function that shuffles the data in two matrices"""
# pylint: disable=invalid-name
import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data in two matrices in the same way
    Arguments:
        X: the first np.ndarray of shape (m, nx) to shuffle, where
           m is the number of data points, and
           nx is the number of features
        Y: the second np.ndarray of shape (m, ny) to shuffle, where
           m is the same number of data points as in X, and
           ny is the number of features in Y
    Return:
        the shuffled matrices
    """
    perm = np.random.permutation(X.shape[0])
    return (X[perm], Y[perm])
