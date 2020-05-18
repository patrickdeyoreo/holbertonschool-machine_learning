#!/usr/bin/env python3
"""Provides a function to perform gradient descent"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """
    Performs gradient descent on the neural network using
    Arguments:
        the path where the model was saved
    """
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)