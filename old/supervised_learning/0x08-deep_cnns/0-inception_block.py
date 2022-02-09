#!/usr/bin/env python3
"""Provides an implementation of an inception inception_block in Keras."""
# pylint: disable=invalid-name

import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    Build an inception block as described in "Going Deeper with Convolutions".

    Arguments:
        A_prev (K.layers.Layer): the output of the previous layer
        filters (Iterable):
            F1, F3R, F3,F5R, F5, and FPP, respectively, where
            F1 is the number of filters in the 1x1 convolution,
            F3R is the number of filters in the 1x1 convolution
            before the 3x3 convolution,
            F3 is the number of filters in the 3x3 convolution,
            F5R is the number of filters in the 1x1 convolution
            before the 5x5 convolution,
            F5 is the number of filters in the 5x5 convolution,
            FPP is the number of filters in the 1x1 convolution
            after the max pooling
    Returns:
        the concatenated output of the inception block
    """
    layers = iter((
        K.layers.Conv2D(
            filters=filters[0], kernel_size=1, padding='same',
            kernel_initializer=K.initializers.he_normal(), activation='relu'
        ),
        K.layers.Conv2D(
            filters=filters[1], kernel_size=1, padding='same',
            kernel_initializer=K.initializers.he_normal(), activation='relu'
        ),
        K.layers.Conv2D(
            filters=filters[2], kernel_size=3, padding='same',
            kernel_initializer=K.initializers.he_normal(), activation='relu'
        ),
        K.layers.Conv2D(
            filters=filters[3], kernel_size=1, padding='same',
            kernel_initializer=K.initializers.he_normal(), activation='relu'
        ),
        K.layers.Conv2D(
            filters=filters[4], kernel_size=5, padding='same',
            kernel_initializer=K.initializers.he_normal(), activation='relu'
        ),
        K.layers.MaxPool2D(
            pool_size=3, padding='same', strides=1
        ),
        K.layers.Conv2D(
            filters=filters[5], kernel_size=1, padding='same',
            kernel_initializer=K.initializers.he_normal(), activation='relu'
        )
    ))
    output = [next(layers)(A_prev)]
    output.extend(next(layers)(layer(A_prev)) for layer in layers)

    return K.layers.concatenate(output)
