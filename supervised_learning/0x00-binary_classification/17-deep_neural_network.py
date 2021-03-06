#!/usr/bin/env python3
"""Provides a class ``DeepNeuralNetwork'' for binary classification"""
# pylint: disable=invalid-name

import numpy as np


class DeepNeuralNetwork:
    """Represents a deep neural network for binary classification"""

    def __init__(self, nx, layers):
        """
        Initializes a binary classification neuron
        Arguments:
            nx: the number of input features
            layers: a list representing the number of nodes in each layer
        """
        if isinstance(nx, int) is False:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if isinstance(layers, list) is False or len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        d1 = nx
        for index, d0 in enumerate(layers, 1):
            if isinstance(d0, int) is False or d0 < 1:
                raise TypeError("layers must be a list of positive integers")
            key = 'W{}'.format(index)
            self.__weights[key] = np.random.randn(d0, d1) * np.sqrt(2 / d1)
            key = 'b{}'.format(index)
            self.__weights[key] = np.zeros((d0, 1))
            d1 = d0

    @property
    def L(self):
        """
        Get the number of layers
        Return:
            the number of layers
        """
        return self.__L

    @property
    def cache(self):
        """
        Get the intermediary values of the network
        Return:
            the cache dictionary
        """
        return self.__cache

    @property
    def weights(self):
        """
        Get the weights and biases of the network
        Return:
            the weights dictionary
        """
        return self.__weights
