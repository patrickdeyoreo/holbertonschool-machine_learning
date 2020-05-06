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
        self.__A = self.sigmoid(self.__W.dot(X) + self.__b)
        return self.__A

    def evaluate(self, X, Y):
        """
        Evaluates the neuron given outputs and targets
        Arguments:
            X: numpy.ndarray with shape (nx, m) that contains the input, where
               nx is the number of input features to the neuron, and
               m is the number of examples
            Y: numpy.ndarray with shape (1, m) that contains the correct labels
        Return:
            the neuron’s prediction and the cost of the network, respectively
        """
        P = np.where(self.forward_prop(X) < 0.5, 0, 1)
        c = self.cost(Y, self.A)
        return (P, c)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates a pass of gradient descent and updates the weights and bias
        Arguments:
            X: numpy.ndarray with shape (nx, m) that contains the input, where
               nx is the number of input features to the neuron, and
               m is the number of examples
            Y: numpy.ndarray with shape (1, m) that contains the correct labels
            A: numpy.ndarray with shape (1, m) containing the activated output
            alpha: the learning rate
        """
        m = X.shape[1]
        dZ = A - Y
        dW = (1 / m) * (X @ dZ.T)
        db = (1 / m) * np.sum(dZ, axis=1)
        self.__W -= alpha * dW[0]
        self.__b -= alpha * db[0]

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
        c = sum(np.log(a) if y == 1 else np.log(1.0000001 - a)
                for y, a in zip(Y[0], A[0]))
        return -c / m

    @staticmethod
    def sigmoid(X):
        """
        Sigmoid activation function
        Arguments:
            X: the x-values
        Return:
            the sigmoid of X
        """
        return 1 / (1 + np.exp(-X))
