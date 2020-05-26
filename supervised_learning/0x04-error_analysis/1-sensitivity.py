#!/usr/bin/env python3
"""Provides a function that calculates sensitivity for a confusion matrix"""
# pylint: disable=invalid-name
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix
    Arguments:
        confusion: numpy.ndarray of shape (classes, classes) where row indices
                   represent the correct labels and column indices represent
                   the predicted labels
    Return:
        np.ndarray of shape (classes,) containing the sensitivity of each class
    """
    TP = np.diagonal(confusion)
    TP_FN = np.sum(confusion, axis=1)
    return TP / TP_FN
