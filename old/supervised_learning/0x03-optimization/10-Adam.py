#!/usr/bin/env python3
"""
Provides a function that creates a training operation for a neural network
in tensorflow using the Adam optimization algorithm
"""
# pylint: disable=invalid-name
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Creates a training operation for a neural network in tensorflow using
    the Adam optimization algorithm
    Arguments:
        loss: the loss of the network
        alpha: the learning rate
        beta1: the weight used for the first moment
        beta2: the weight used for the second moment
        epsilon: a small number to avoid division by zero
    Return:
        the Adam optimization operation
    """
    optimizer = tf.train.AdamOptimizer(
        learning_rate=alpha, beta1=beta1, beta2=beta2, epsilon=epsilon)
    return optimizer.minimize(loss)
