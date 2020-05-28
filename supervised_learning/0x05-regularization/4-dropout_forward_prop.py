#!/usr/bin/env python3
"""
Provides a function to perform forward propagation using Dropout
"""
# pylint: disable=invalid-name,too-many-arguments
import numpy as np


def softmax(X):
    """
    softmax activation
    Arguments:
        X: a np.ndarray containing the activation input
    Return:
        a np.ndarray containing the activated output
    """
    logits = np.exp(X - np.max(X))
    return logits / np.sum(logits, axis=0, keepdims=True)


def tanh(X):
    """
    tanh activation
    Arguments:
        X: a np.ndarray containing the activation input
    Return:
        a np.ndarray containing the activated output
    """
    e_pos = np.exp(X)
    e_neg = np.exp(-X)
    return (e_pos - e_neg) / (e_pos + e_neg)


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Performs forward propagation of a neural network using Dropout
    All layers except the last use the tanh activation function
    The last layer should uses softmax activation function
    Arguments:
        X: a numpy.ndarray of shape (nx, m) containing the input data, where
           nx is the number of input features, and
           m is the number of data points
        weights: a dictionary of the weights and biases of the neural network
        L: the number of layers in the network
        keep_prob: the probability that a node will be kept
    Return:
        a dictionary containing the output and dropout mask of each layer
    """
    outputs = {}
    A = outputs['A0'] = X
    for layer in range(1, L):
        W = weights['W{}'.format(layer)]
        b = weights['b{}'.format(layer)]
        D = outputs['D{}'.format(layer)] = np.random.binomial(
            1, keep_prob, size=(W.shape[0], A.shape[1]))
        A = outputs['A{}'.format(layer)] = tanh(W @ A + b) * D / keep_prob
    W = weights['W{}'.format(L)]
    b = weights['b{}'.format(L)]
    A = outputs['A{}'.format(L)] = softmax(W @ A + b)
    return outputs
