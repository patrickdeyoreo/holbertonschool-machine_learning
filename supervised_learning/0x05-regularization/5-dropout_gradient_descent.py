#!/usr/bin/env python3
"""
Provides a function to update the weights and biases of a neural network using
gradient descent with Dropout regularization
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


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights and biases of a neural network by performing
    gradient descent with Dropout regularization

    The neural network uses tanh activations on each layer except the last,
    which uses a softmax activation

    Arguments:
        Y: a one-hot np.ndarray of shape (classes, m) of correct labels, where
           classes is the number of classes, and
           m is the number of data points
        weights: a dictionary of the weights and biases of the neural network
        cache: a dictionary of the output of each layer of the network
        alpha: the learning rate
        keep_prob: the probability that a node will be kept
        L: the number of layers in the network
    """
    m = Y.shape[1]
    A = cache['A{}'.format(L)]
    dZ = A - Y
    for layer in reversed(range(1, L)):
        D = cache['D{}'.format(layer)]
        A = cache['A{}'.format(layer)]
        W = weights['W{}'.format(layer + 1)]
        dW = dZ @ A.T / keep_prob / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dZ = W.T @ dZ * tanh_prime(A) * D
        weights['W{}'.format(layer + 1)] -= alpha * dW
        weights['b{}'.format(layer + 1)] -= alpha * db
    A = cache['A0']
    dW = dZ @ A.T / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    weights['W1'] -= alpha * dW
    weights['b1'] -= alpha * db
