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

    def evaluate(self, X, Y):
        """
        Evaluate the neuron’s predictions.
        Args:
            X (numpy.ndarray):
                array with shape (nx, m) containing the input data, where nx is
                the number of input features to the neuron, and m is the number
                of examples
            Y (numpy.ndarray):
                array with shape (1, m) that contains the correct labels for
                the input data
        Returns:
            (numpy.ndarray):
                the neuron's prediction as an array with shape (1, m)
                containing the predicted labels for each example
            (float):
                the cost of the model
        """
        A = self.forward_prop(X)
        Z = np.where(A < 0.5, 0, 1)
        c = self.cost(Y, A)
        return Z, c

    def forward_prop(self, X):
        """
        Compute forward propagation.
        Args:
            X (numpy.ndarray):
                array with shape (nx, m) containing the input data, where nx is
                the number of input features to the neuron, and m is the number
                of examples
        Returns:
            (numpy.ndarray):
                array of shape (1, m) containing the activation state of the
                neuron for each example, where m is the number of examples
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
                input data, where m is the number of examples
            A (numpy.ndarray):
                array of shape (1, m) containing the activation state of the
                neuron for each example, where m is the number of examples
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
                weights and the input data, where m is the number of examples
        Returns:
            (numpy.ndarray):
                array of shape (1, m) containing the activation state of the
                neuron for each example, where m is the number of examples
        """
        return (1 + np.exp(-X)) ** (-1)
