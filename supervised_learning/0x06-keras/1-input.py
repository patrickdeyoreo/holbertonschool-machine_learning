#!/usr/bin/env python3
"""
Provides a function that builds a neural network with the Keras library
"""
# pylint: disable=invalid-name
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library
    Arguments:
        nx: the number of input features to the network
        layers: a list containing the number of nodes in each layer
        activations: a list containing the activation functions for each layer
        lambtha: the L2 regularization parameter
        keep_prob: the probability that a node will be kept for dropout
    Return:
        the keras model
    """
    outputs = inputs = K.Input(shape=(nx,))
    items = zip(layers, activations)
    first = next(items, None)
    if first is not None:
        kwgs = dict(kernel_regularizer=K.regularizers.l2(lambtha))
        units, kwgs['activation'] = first
        outputs = K.layers.Dense(units, **kwgs)(outputs)
    for units, kwgs['activation'] in items:
        outputs = K.layers.Dropout(1 - keep_prob)(outputs)
        kwgs.update(kernel_regularizer=K.regularizers.l2(lambtha))
        outputs = K.layers.Dense(units, **kwgs)(outputs)
    return K.Model(inputs=inputs, outputs=outputs)
