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
          nx (int): the number of inputs to the neuron
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
        Get the weights vector of the neuron.
        Returns:
            (numpy.ndarray): the weights vector of the neuron
        """
        return self.__W

    @property
    def b(self):
        """
        Get the bias of the neuron.
        Returns:
            (int): the bias the neuron
        """
        return self.__b

    @property
    def A(self):
        """
        Get the activation state of the neuron.
        Returns:
            (Union[numpy.ndarray, int]): the activation state of the neuron
        """
        return self.__A

    def forward_prop(self, X):
        """
        Compute forward propagation.
        Args:
            X (numpy.ndarray):
                array with shape (nx, m) containing the input data, where nx
                is the number of input features to the neuron, and m is the
                number of samples
        Returns:
            (numpy.ndarray):
                array of shape (1, m) containing the activation state of the
                neuron for each sample, where m is the number of samples
        """
        self.__A = self.sigmoid(self.__W @ X + self.__b)
        return self.__A

    @staticmethod
    def cost(Y, A):
        """
        Compute the cost of the model using logistic regression.
        Args:
            Y (numpy.ndarray):
                array with shape (1, m) containing the correct labels for the
                input data, where m is the number of samples
            A (numpy.ndarray):
                array of shape (1, m) containing the activation state of the
                neuron for each sample, where m is the number of samples
        Returns:
            (float): the cost of the model
        """
        m = Y.shape[1]
        c1 = (-1) * np.log(A) * Y
        c0 = (-1) * np.log(1.0000001 - A) * (1 - Y)
        return np.sum(c1 + c0) / m

    @staticmethod
    def sigmoid(X):
        """
        Compute the sigmoid activation.
        Args:
            X (numpy.ndarray):
                array of shape (1, m) containing the cross product of the
                weights and the input data, where m is the number of samples
        Returns:
            (numpy.ndarray):
                array of shape (1, m) containing the activation state of the
                neuron for each sample, where m is the number of samples
        """
        return (1 + np.exp(-X)) ** (-1)
