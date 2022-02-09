#!/usr/bin/env python3
"""Provides a function to define gradient descent for a neural network"""
# pylint: disable=invalid-name

import tensorflow as tf


def create_train_op(loss, alpha):
    """
    Defines an operation to perform gradient descent
    Arguments:
        the path where the model was saved
    Return:
        an operation that trains the network using gradient descent
    """
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
