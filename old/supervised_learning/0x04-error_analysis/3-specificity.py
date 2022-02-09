#!/usr/bin/env python3
"""Provides a function that calculates specificity for a confusion matrix"""
# pylint: disable=invalid-name
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix
    Arguments:
        confusion: numpy.ndarray of shape (classes, classes) where row indices
                   represent the correct labels and column indices represent
                   the predicted labels
    Return:
        np.ndarray of shape (classes,) containing the specificity of each class
    """
    total = np.sum(confusion)
    TP = np.diagonal(confusion)
    TP_FP = np.sum(confusion, axis=0)
    TP_FN = np.sum(confusion, axis=1)
    FP = TP_FP - TP
    TN = total - FP - TP_FN
    return TN / (TN + FP)
