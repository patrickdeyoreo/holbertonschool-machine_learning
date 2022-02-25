#!/usr/bin/env python3
"""
Provide a class Neuron that defines a neuron for binary classification.
"""
import numpy as np


class Neuron:
    """
    Define a neuron for performing binary classification.
    """
    def __init__(self, nx):
        """
        Initialize a neuron.
        Args:
          nx (int): number of inputs to the neuron
        Raises:
          TypeError: argument 'nx' is not an integer
          ValueError: argument 'nx' is less than 1
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
