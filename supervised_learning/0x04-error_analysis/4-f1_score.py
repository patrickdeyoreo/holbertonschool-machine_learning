#!/usr/bin/env python3
"""Provides a function that calculates the F1 score of a confusion matrix"""
# pylint: disable=invalid-name
import numpy as np

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score of a confusion matrix
    Arguments:
        confusion: numpy.ndarray of shape (classes, classes) where row indices
                   represent the correct labels and column indices represent
                   the predicted labels
    Return:
        np.ndarray of shape (classes,) containing the F1 score of each class
    """
    TP = np.diagonal(confusion)
    TP_FP = np.sum(confusion, axis=0)
    TP_FN = np.sum(confusion, axis=1)
    return 2 * TP / (TP_FP + TP_FN)
