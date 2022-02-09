#!/usr/bin/env python3
"""
Provides a function that normalizes an unactivated output of a neural network
using batch normalization
"""
# pylint: disable=invalid-name,too-many-arguments


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes unactivated output of a neural network using batch normalization
    Arguments:
        Z: np.ndarray of shape (m, n) that should be normalized, where
           m is the number of data points, and
           n is the number of features in Z
        gamma: np.ndarray of shape (1, n) of the scales for normalization
        beta: np.ndarray of shape (1, n) of the offsets for normalization
        epsilon: a small number used to avoid division by zero
    Return:
        the normalized Z matrix
    """
    mean = Z.mean(axis=0)
    var = Z.var(axis=0)
    stddev = (var + epsilon) ** 0.5
    center = Z - mean
    normal = center / stddev
    return gamma * normal + beta
