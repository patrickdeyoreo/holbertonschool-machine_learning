#!/usr/bin/env python3
"""
Provides a function that updates a variable using the RMSProp algorithm
"""
# pylint: disable=invalid-name,too-many-arguments


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm
    Arguments:
        alpha: the learning rate
        beta2: the RMSProp weight
        epsilon: a small number to avoid division by zero
        var: np.ndarray containing the variable to be updated
        grad: np.ndarray containing the gradient of var
        s: the previous second moment of var
    Return:
        the updated variable and the new moment, respectively
    """
    s *= beta2
    s += (1 - beta2) * grad ** 2
    var -= alpha * grad / (s ** 0.5 + epsilon)
    return (var, s)
