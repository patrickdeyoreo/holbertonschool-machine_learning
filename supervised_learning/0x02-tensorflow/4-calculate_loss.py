#!/usr/bin/env python3
"""Provides a function to calculate loss"""

import tensorflow as tf


def calculate_loss(y, pred):
    """
    Calculates softmax loss
    Arguments:
        y: the correct labels
        pred: the predicted labels
    Return:
        a tensor containing the loss of the prediction
    """
    return tf.losses.softmax_cross_entropy(logits=pred, onehot_labels=y)