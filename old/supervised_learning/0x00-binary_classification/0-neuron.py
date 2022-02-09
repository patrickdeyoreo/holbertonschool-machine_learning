#!/usr/bin/env python3
"""Provides a class ``Neuron'' to represent a binary classification neuron"""
# pylint: disable=invalid-name

import numpy as np


class Neuron:
    """Represents a binary classification Neuron"""
    # pylint: disable=too-few-public-methods

    def __init__(self, nx):
        """
        Initializes a binary classification neuron
        Arguments:
            nx: the number of input features
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
