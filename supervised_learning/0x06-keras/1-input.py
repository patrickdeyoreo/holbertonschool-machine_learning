#!/usr/bin/env python3
"""
Provides a function that builds a neural network with the Keras library
"""
# pylint: disable=import-error,invalid-name
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
    inputs = K.Input(shape=(nx,))
    pairs = zip(layers, activations)
    units, activation = next(pairs)
    kwgs = dict(activation=activation,
                kernel_regularizer=K.regularizers.l2(lambtha))
    outputs = K.layers.Dense(units, **kwgs)(inputs)
    for units, activation in pairs:
        outputs = K.layers.Dropout(1 - keep_prob)(outputs)
        kwgs.update(activation=activation,
                    kernel_regularizer=K.regularizers.l2(lambtha))
        outputs = K.layers.Dense(units, **kwgs)(outputs)
    return K.Model(inputs=inputs, outputs=outputs)
