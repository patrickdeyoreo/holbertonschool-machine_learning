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
    TP = np.diagonal(confusion)
    P = np.sum(confusion, axis=0)
    TP_FN = np.sum(confusion, axis=1)
    FP = P - TP
    TP_F = TP_FN + FP
    TN = np.sum(confusion) - TP_F
    return TN / (TN + FP)
