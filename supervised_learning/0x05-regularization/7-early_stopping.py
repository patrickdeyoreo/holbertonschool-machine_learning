#!/usr/bin/env python3
"""
Provides a function to determine if gradient descent should stop early
"""
# pylint: disable=invalid-name,too-many-arguments
import numpy as np


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if gradient descent should stop early

    Early stopping should occur when the validation cost of the network has not
    decreased relative to the optimal validation cost by more than the
    threshold over a specific patience count

    Arguments:
        cost: the current validation cost of the neural network
        opt_cost: the lowest recorded validation cost of the neural network
        threshold: the threshold used for early stopping
        patience: the patience count used for early stopping
        count: the count of how long the threshold has not been met
    Return:
        a boolean stating wether or not gradient descent should stop early, and
        an updated count
    """
    if opt_cost - cost <= threshold:
        return (patience <= count + 1, count + 1)
    return (False, 0)
