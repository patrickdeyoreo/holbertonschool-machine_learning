#!/usr/bin/env python3
"""
Provides a function that updates a variable using the Adam algorithm
"""
# pylint: disable=invalid-name,too-many-arguments
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable using the Adam optimization algorithm
    Arguments:
        alpha: the learning rate
        beta1: the weight used for the first moment
        beta2: the weight used for the second moment
        epsilon: a small number to avoid division by zero
        var: a numpy.ndarray containing the variable to be updated
        grad: a numpy.ndarray containing the gradient of var
        v: the previous first moment of var
        s: the previous second moment of var
        t: the time step used for bias correction
    Return:
        the updated variable, the new first moment, and the new second moment
    """
    v *= beta1
    v += (1 - beta1) * grad
    s *= beta2
    s += (1 - beta2) * grad ** 2
    v_b = v / (1 - beta1 ** t)
    s_b = s / (1 - beta2 ** t)
    var -= alpha * v_b / (np.sqrt(s_b) + epsilon)
    return (var, v_b, s_b)
