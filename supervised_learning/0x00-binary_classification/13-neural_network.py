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

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the network using a sigmoid
        activation function and updates the activation state
        Arguments:
            X: numpy.ndarray with shape (nx, m) that contains the input, where
               nx is the number of input features to the neuron, and
               m is the number of examples
        Return:
            the activation state
        """
        self.__A1 = self.sigmoid(self.W1 @ X + self.b1)
        self.__A2 = self.sigmoid(self.W2 @ self.A1 + self.b2)
        return (self.A1, self.A2)

    def evaluate(self, X, Y):
        """
        Evaluates performance of the network with the given outputs and targets
        Arguments:
            X: numpy.ndarray with shape (nx, m) that contains the input, where
               nx is the number of input features to the neuron, and
               m is the number of examples
            Y: numpy.ndarray with shape (1, m) that contains the correct labels
        Return:
            the neuron’s prediction and the cost of the network, respectively
        """
        _, A2 = self.forward_prop(X)
        P = np.where(A2 < 0.5, 0, 1)
        c = self.cost(Y, A2)
        return (P, c)

    # pylint: disable=too-many-arguments
    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates a pass of gradient descent and updates the weights and bias
        Arguments:
            X: numpy.ndarray with shape (nx, m) that contains the input, where
               nx is the number of input features to the neuron, and
               m is the number of examples
            Y: numpy.ndarray with shape (1, m) that contains the correct labels
            A1: the output of the hidden layer
            A2: the predicted output
            alpha: the learning rate
        """
        m = X.shape[1]
        dZ2 = A2 - Y
        dW2 = (1 / m) * (dZ2 @ A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = (self.W2.T @ dZ2) * (A1 * (1 - A1))
        dW1 = (1 / m) * (dZ1 @ X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1
        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2

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
    def sigmoid(X):
        """
        Sigmoid activation function
        Arguments:
            X: the x-values
        Return:
            the sigmoid of X
        """
        return 1 / (1 + np.exp(-X))
