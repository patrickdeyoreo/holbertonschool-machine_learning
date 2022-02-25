#!/usr/bin/env python3
"""
Provide a class Neuron that defines a neuron for binary classification.
"""
from tempfile import gettempprefix
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
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        Get the attribute 'W'
        """
        return self.__W

    @property
    def b(self):
        """
        Get the attribute 'b'
        """
        return self.__b

    @property
    def A(self):
        """
        Get the attribute 'A'
        """
        return self.__A
