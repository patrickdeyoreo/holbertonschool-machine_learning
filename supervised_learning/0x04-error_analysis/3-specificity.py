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
    classes = confusion.shape[0]
    tf_all = confusion.sum()
    tf_pos = confusion.sum(axis=0)
    t_pos_f_neg = confusion.sum(axis=1)
    f_pos = tf_pos - np.array([confusion[n][n] for n in range(classes)])
    tf_pos_f_neg = t_pos_f_neg + f_pos
    t_neg = np.array([tf_all - tf_pos_f_neg[n] for n in range(classes)])
    return np.array([t_neg[n] / (t_neg[n] + f_pos[n]) for n in range(classes)])
