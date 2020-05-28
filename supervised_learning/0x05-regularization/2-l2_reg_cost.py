#!/usr/bin/env python3
"""
Provides a function to calculate the regularized cost of a neural network
"""
# pylint: disable=invalid-name,too-many-arguments
import tensorflow as tf


def l2_reg_cost(cost):
    """
    Calculates the cost of a neural network with L2 regularization
    Arguments:
        cost: a tensor containing the cost of the network before regularization
    Return:
        a tensor containing the cost of the network after L2 regularization
    """
    return cost + tf.losses.get_regularization_losses()
