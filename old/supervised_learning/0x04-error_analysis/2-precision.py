#!/usr/bin/env python3
"""Provides a function that calculates precision for a confusion matrix"""
# pylint: disable=invalid-name
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix
    Arguments:
        confusion: numpy.ndarray of shape (classes, classes) where row indices
                   represent the correct labels and column indices represent
                   the predicted labels
    Return:
        np.ndarray of shape (classes,) containing the precision of each class
    """
    TP = np.diagonal(confusion)
    TP_FP = np.sum(confusion, axis=0)
    return TP / TP_FP
