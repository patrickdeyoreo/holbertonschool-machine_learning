#!/usr/bin/env python3
"""Provides a function to calculate the accuracy of a prediction"""
# pylint: disable=invalid-name

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction
    Arguments:
        y: a placeholder for the labels of the input
        y_pred: the predictions
    Return:
        a tensor containing the accuracy of the prediction
    """
    return tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1)), tf.float32))
