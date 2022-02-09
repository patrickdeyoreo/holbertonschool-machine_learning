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

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron using a sigmoid
        activation function and updates the activation state
        Arguments:
            X: numpy.ndarray with shape (nx, m) that contains the input, where
               nx is the number of input features to the neuron, and
               m is the number of examples
        Return:
            the activation state
        """
        self.__A = self.sigmoid(self.__W @ X + self.__b)
        return self.__A

    @staticmethod
    def cost(Y, A):
        """
        Calculates the cost of the model using logistic regression
        Arguments:
            Y: numpy.ndarray with shape (1, m) that contains the correct labels
            A: numpy.ndarray with shape (1, m) containing the activated output
        Return:
            the cost
        """
        m = Y.shape[1]
        c1 = np.log(A) * Y
        c0 = np.log(1.0000001 - A) * (1 - Y)
        return (-1 / m) * np.sum(c1 + c0)

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid activation function
        Arguments:
            x: the x-value
        Return:
            the activation state
        """
        return (1 + np.exp(-x)) ** (-1)
