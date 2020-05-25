#!/usr/bin/env python3
"""Provides a function to normalize a matrix"""
# pylint: disable=invalid-name


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix
    Arguments:
        X: np.ndarray with shape (d, nx), where
           d is the number of data points, and
           nx is the number of features
        m: np.ndarray with shape (nx,) of the mean of all features of X
        s: np.ndarray with shape (nx,) of the stddev of all features of X
    Return:
        the normalized X matrix
    """
    return (X - m) / s
