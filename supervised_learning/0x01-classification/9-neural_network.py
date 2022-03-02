#!/usr/bin/env python3
"""
Provide a class NeuralNetwork that defines a neural network with one hidden
layer for performing binary classification.
"""
import numpy as np


class NeuralNetwork:
    """
    Define a neural network for performing binary classification.
    """
    def __init__(self, nx, nodes):
        """
        Initialize a neural network.
        Args:
          nx (int):
            the number of inputs to the neuron
          nodes(int):
            the number of nodes in the network
        Raises:
          TypeError:
            argument 'nx' is not an integer, or argument 'nodes' is not an
            integer
          ValueError:
            argument 'nx' is less than 1, or argument 'nodes' is less than 1
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
    def A1(self):
        """
        Get the activation state of the hidden layer.
        Return:
            (numpy.ndarray): the activation state of the hidden layer
        """
        return self.__A1

    @property
    def W1(self):
        """
        Get the weights vector of the hidden layer.
        Returns:
            (numpy.ndarray): the weights vector of the hidden layer
        """
        return self.__W1

    @property
    def b1(self):
        """
        Get the bias vector of the hidden layer.
        Return:
            (numpy.ndarray): the bias vector of the hidden layer
        """
        return self.__b1

    @property
    def A2(self):
        """
        Get the activation state of the output layer.
        Return:
            (numpy.ndarray): the activation state of the output layer
        """
        return self.__A2

    @property
    def W2(self):
        """
        Get the weights vector of the output layer.
        Returns:
            (numpy.ndarray): the weights vector of the output layer
        """
        return self.__W2

    @property
    def b2(self):
        """
        Get the bias vector of the output layer.
        Return:
            (numpy.ndarray): the bias vector of the output layer
        """
        return self.__b2
