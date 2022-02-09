#!/usr/bin/env python3
"""Provides a class ``NeuralNetwork'' for binary classification"""
# pylint: disable=invalid-name

import numpy as np


class NeuralNetwork:
    """Represents a binary classification network with one hidden layer"""

    def __init__(self, nx, nodes):
        """
        Initializes a binary classification neuron
        Arguments:
            nx: the number of input features
            nodes: the number of nodes found in the hidden layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        Get the weights vector of the hidden layer
        Return:
            the weights vector
        """
        return self.__W1

    @property
    def b1(self):
        """
        Get the bias vector of the hidden layer
        Return:
            the bias
        """
        return self.__b1

    @property
    def A1(self):
        """
        Get the activation state of the hidden layer
        Return:
            the activation state
        """
        return self.__A1

    @property
    def W2(self):
        """
        Get the weights vector of the output layer
        Return:
            the weights vector
        """
        return self.__W2

    @property
    def b2(self):
        """
        Get the bias vector of the output layer
        Return:
            the bias
        """
        return self.__b2

    @property
    def A2(self):
        """
        Get the activation state of the output layer
        Return:
            the activation state
        """
        return self.__A2
