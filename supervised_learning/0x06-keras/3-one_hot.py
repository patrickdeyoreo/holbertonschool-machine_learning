#!/usr/bin/env python3
"""
Provides a function to convert a vector of labels into a one-hot matrix
"""
# pylint: disable=invalid-name
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a vector of labels into a one-hot matrix
    Arguments:
        labels: the class labels
        classes: the total number of classes
    Return:
        a one-hot encoding of the input
    """
    return K.utils.to_categorical(labels, num_classes=classes)
