#!/usr/bin/env python3
"""
Provides a function to calculate the regularized cost of a neural network
"""
# pylint: disable=invalid-name
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization
    Arguments:
        cost: the cost of the network without L2 regularization
        lambtha: the regularization parameter
        weights: a dictionary of the weights and biases of the neural network
        L: the number of layers in the neural network
        m: the number of data points used
    Return:
        the cost of the network accounting for L2 regularization
    """
    R = sum(np.linalg.norm(weights['W{}'.format(i)]) for i in range(1, L + 1))
    return cost + lambtha * R / (2 * m)
