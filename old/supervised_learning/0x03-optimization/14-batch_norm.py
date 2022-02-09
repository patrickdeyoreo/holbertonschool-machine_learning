#!/usr/bin/env python3
"""
Provides a function that creates a batch norm layer for a neural network
in tensorflow
"""
# pylint: disable=invalid-name,too-many-arguments
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow
    Arguments:
        prev: the activated output of the previous layer
        n: the number of nodes in the layer to be created
        activation: the activation function to use on the output of the layer
    Return:
        a tensor of the activated output for the layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG')
    layer = tf.layers.Dense(
        units=n, activation=None, kernel_initializer=init, name='layer')
    mean, variance = tf.nn.moments(layer(prev), axes=[0])
    offset = tf.Variable(tf.zeros([n]), trainable=True, name='beta')
    scale = tf.Variable(tf.ones([n]), trainable=True, name='gamma')
    norm = tf.nn.batch_normalization(
        layer(prev), mean, variance, offset, scale, variance_epsilon=1e-8)
    return activation(norm)
