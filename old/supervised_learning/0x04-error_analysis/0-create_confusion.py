#!/usr/bin/env python3
"""Provides a function that creates a confusion matrix"""
# pylint: disable=invalid-name
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix
    Arguments:
        labels: one-hot numpy.ndarray of shape (m, classes) containing the
                correct labels for each data point, where
                m is the number of data points, and
                classes is the number of classes
        logits: one-hot numpy.ndarray of shape (m, classes) containing the
                predicted labels
    Return:
        confusion numpy.ndarray of shape (classes, classes) with row indices
        representing the correct labels and column indices representing the
        predicted labels
    """
    confusion = np.zeros((labels.shape[1], labels.shape[1]))
    for true, pred in zip(labels, logits):
        confusion[true.argmax()][pred.argmax()] += 1
    return confusion
