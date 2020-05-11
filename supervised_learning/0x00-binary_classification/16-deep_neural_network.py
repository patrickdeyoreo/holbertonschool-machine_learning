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
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(item, int) and item > 0 for item in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = dict(
            item for idx, dim in
            enumerate(zip(layers, [nx] + layers), 1) for item in
            (('W{}'.format(idx), np.random.randn(*dim) * (2 / dim[1]) ** 0.5),
             ('b{}'.format(idx), np.zeros((dim[0], 1)))))
