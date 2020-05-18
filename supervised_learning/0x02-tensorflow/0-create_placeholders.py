#!/usr/bin/env python3
"""Provides a function to create two placeholders for a neural network"""
# pylint: disable=invalid-name

import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Creates two placeholders for a neural network
    Arguments:
        nx: the number of feature columns in our data
        classes: the number of classes in our classifier
    Returns:
        x: the placeholder for the input data to the neural network
        y: the placeholder for the one-hot labels for the input data
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    return (x, y)
