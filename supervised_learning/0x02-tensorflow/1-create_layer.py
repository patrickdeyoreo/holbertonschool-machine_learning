#!/usr/bin/env python3
"""Provides a function to create a layer for a neural network"""
# pylint: disable=invalid-name

import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Creates a layer for a neural network
    Arguments:
        prev: the tensor output of the previous layer
        n: the number of nodes in the layer to create
        activation: the activation function
    Returns:
        the tensor output of the layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG')
    layer = tf.layers.Dense(
        units=n, activation=activation, kernel_initializer=init, name='layer')
    return layer(prev)
