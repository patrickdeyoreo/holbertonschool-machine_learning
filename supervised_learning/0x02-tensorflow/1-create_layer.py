#!/usr/bin/env python3
"""Provides q function to create a layer"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Creates a layer for q neural network
    Arguments:
        prev: the previous layer
        n: the number of nodes
        activation: the activation function
    """
    index = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(
        units=n, activation=activation, kernel_initializer=index, name="layer")
    return layer(prev)
