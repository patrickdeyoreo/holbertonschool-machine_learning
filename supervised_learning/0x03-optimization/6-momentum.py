#!/usr/bin/env python3
"""
Provides a function that creates a training operation for a neural network
in tensorflow using gradient descent with a momentum optimization algorithm
"""
# pylint: disable=invalid-name
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    Creates a training operation for a neural network in tensorflow using
    gradient descent with a momentum optimization algorithm
    Arguments:
        loss: the loss of the network
        alpha: the learning rate
        beta1: the momentum weight
    Return:
        the momentum optimization operation
    """
    optimizer = tf.train.MomentumOptimizer(learning_rate=alpha, momentum=beta1)
    return optimizer.minimize(loss)
