#!/usr/bin/env python3
"""Provides a function to calculate normalization constants"""
# pylint: disable=invalid-name
import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization constants of a matrix
    Arguments:
        X: np.ndarray of shape (m, nx) to normalize, where
           m is the number of data points, and
           nx is the number of features
    Return:
        the mean and standard deviation of each feature, respectively
    """
    return (np.mean(X, axis=0), np.std(X, axis=0))
