#!/usr/bin/env python3
"""Provides a function to perform forward propagation"""
# pylint: disable=invalid-name

import tensorflow as tf

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Perform forward propagation in a neural network.
    Arguments:
        x: the placeholder for input
        layer_sizes: a list of the number of nodes in each layer
        activations: a list of activation functions to use
    Return:
        the prediction of the network in tensor form
    """
    y_pred = x
    for size, activation in zip(layer_sizes, activations):
        y_pred = create_layer(y_pred, size, activation)
    return y_pred
