#!/usr/bin/env python3
"""Provides a function to compute accuracy"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """uses tf.reduce_mean, y is the lables, y_pred is the predictions"""
    label = tf.argmax(y, 1)
    pred = tf.argmax(pred, 1)
    equal = tf.equal(pred, label)
    return tf.reduce_mean(tf.cast(equal, tf.float32))