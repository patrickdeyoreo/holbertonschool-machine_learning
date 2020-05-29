#!/usr/bin/env python3
"""
Provides a function to create a tensorflow layer with L2 regularization
"""
# pylint: disable=invalid-name,too-many-arguments
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a tensorflow layer with L2 regularization
    Arguments:
        prev: a tensor containing the output of the previous layer
        n: the number of nodes the new layer should contain
        activation: the activation function that should be used on the layer
        lambtha: the L2 regularization parameter
    Return:
        the output of the new layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG')
    l2_r = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=init,
                            kernel_regularizer=l2_r,
                            name='layer')
    return layer(prev)
