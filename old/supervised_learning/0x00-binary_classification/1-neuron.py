#!/usr/bin/env python3
"""Provides a class ``Neuron'' to represent a binary classification neuron"""
# pylint: disable=invalid-name

import numpy as np


class Neuron:
    """Represents a binary classification Neuron"""

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
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        Get the weights vector of a neuron
        Return:
            the weights vector
        """
        return self.__W

    @property
    def b(self):
        """
        Get the bias of a neuron
        Return:
            the bias
        """
        return self.__b

    @property
    def A(self):
        """
        Get the activation state of a neuron
        Return:
            the activation state
        """
        return self.__A
