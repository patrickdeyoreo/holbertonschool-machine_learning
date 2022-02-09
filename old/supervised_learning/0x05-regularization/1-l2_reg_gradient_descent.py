#!/usr/bin/env python3
"""
Provides a function to update the weights and biases of a neural network using
gradient descent with L2 regularization
"""
# pylint: disable=invalid-name,too-many-arguments
import numpy as np


def tanh_prime(A):
    """
    Computes derivatives of the tanh activation function
    Arguments:
        A: a np.ndarray containing the activated outputs
    Return:
        a np.ndarray containing the derivatives of the activated outputs
    """
    return 1 - A ** 2


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network by performing
    gradient descent with L2 regularization
    The neural network uses tanh activations on each layer except the last,
    which uses a softmax activation
    Arguments:
        Y: a one-hot np.ndarray of shape (classes, m) of correct labels, where
           classes is the number of classes, and
           m is the number of data points
        weights: a dictionary of the weights and biases of the network
        cache: a dictionary of the output of each layer of the network
        alpha: the learning rate
        lambtha: the L2 regularization parameter
        L: the number of layers in the network
        m: the number of data points used
    """
    m = Y.shape[1]
    dZ = cache['A{}'.format(L)] - Y
    for i in reversed(range(L)):
        A = cache['A{}'.format(i)]
        W = weights['W{}'.format(i+1)]
        dW = dZ @ A.T / m + lambtha * W / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dZ = W.T @ dZ * tanh_prime(A)
        weights['W{}'.format(i+1)] -= alpha * dW
        weights['b{}'.format(i+1)] -= alpha * db
