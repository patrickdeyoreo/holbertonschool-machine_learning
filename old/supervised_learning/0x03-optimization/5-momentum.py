#!/usr/bin/env python3
"""
Provides a function that updates a variable using gradient descent with a
momentum optimization algorithm
"""
# pylint: disable=invalid-name


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using gradient descent with a momentum optimization
    algorithm
    Arguments:
        alpha: the learning rate
        beta1: the momentum weight
        var: np.ndarray containing the variable to be updated
        grad: np.ndarray containing the gradient of var
        v: the previous first moment of var
    Return:
        the updated variable and the new moment, respectively
    """
    v *= beta1
    v += (1 - beta1) * grad
    var -= alpha * v
    return (var, v)
