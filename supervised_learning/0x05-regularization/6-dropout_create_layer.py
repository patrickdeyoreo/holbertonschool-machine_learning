#!/usr/bin/env python3
"""
Provides a function to create a tensorflow layer using dropout
"""
# pylint: disable=invalid-name,too-many-arguments
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    Creates a tensorflow layer using dropout
    Arguments:
        prev: a tensor containing the output of the previous layer
        n: the number of nodes the new layer should contain
        activation: the activation function that should be used on the layer
        keep_prob: the probability that a node will be kept
    Return:
        the output of the new layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(
        mode='FAN_AVG')
    layer = tf.layers.Dense(
        units=n, activation=activation, kernel_initializer=init, name='layer')
    dropout = tf.layers.Dropout(
        rate=1-keep_prob, name='dropout')
    return dropout(layer(prev))
