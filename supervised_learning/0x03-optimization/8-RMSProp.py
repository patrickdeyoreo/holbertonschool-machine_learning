#!/usr/bin/env python3
"""
Provides a function that creates a training operation for a neural network
in tensorflow using the RMSProp optimization algorithm
"""
# pylint: disable=invalid-name
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Creates a training operation for a neural network in tensorflow using
    the RMSProp optimization algorithm
    Arguments:
        loss: the loss of the network
        alpha: the learning rate
        beta2: the RMSProp weight
        epsilon: a small number to avoid division by zero
    Return:
        the RMSProp optimization operation
    """
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=alpha, decay=beta2, epsilon=epsilon)
    return optimizer.minimize(loss)
