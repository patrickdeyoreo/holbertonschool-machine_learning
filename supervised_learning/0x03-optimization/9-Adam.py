#!/usr/bin/env python3
"""
Provides a function that updates a variable using the Adam algorithm
"""
# pylint: disable=invalid-name,too-many-arguments


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
    v = v * beta1 + (1 - beta1) * grad
    s = s * beta2 + (1 - beta2) * grad ** 2
    v_t = v / (1 - beta1 ** t)
    s_t = s / (1 - beta2 ** t)
    var -= alpha * (v_t / (s_t ** 0.5 + epsilon))
    return (var, v, s)
