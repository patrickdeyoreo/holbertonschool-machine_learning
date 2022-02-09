#!/usr/bin/env python3
"""Provides a function to calculate loss"""
# pylint: disable=invalid-name

import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    Calculates softmax loss
    Arguments:
        y: the correct labels
        y_pred: the predicted labels
    Return:
        a tensor containing the loss of the prediction
    """
    return tf.losses.softmax_cross_entropy(logits=y_pred, onehot_labels=y)
