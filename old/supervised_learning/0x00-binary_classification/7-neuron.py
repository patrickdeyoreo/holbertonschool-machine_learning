#!/usr/bin/env python3
"""Provides a class ``Neuron'' to represent a binary classification neuron"""
# pylint: disable=invalid-name

import matplotlib.pyplot as plt
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
        dZ = A - Y
        dW = (X @ dZ.T) / X.shape[1]
        db = np.sum(dZ) / X.shape[1]
        self.__W -= (alpha * dW).T
        self.__b -= (alpha * db).T

    # pylint: disable=too-many-arguments,too-many-branches
    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains a neuron
        Arguments:
            X: numpy.ndarray with shape (nx, m) that contains the input, where
               nx is the number of input features to the neuron, and
               m is the number of examples
            Y: numpy.ndarray with shape (1, m) that contains the correct labels
            iterations: the number of iterations to train over
            alpha: the learning rate
            verbose: whether or not to print training information
            graph: whether or not to plot training information upon completion
            step: the granularity of the training information
        Return:
            evaluation of the training data following completion of training
        """
        if isinstance(iterations, int) is False:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if isinstance(alpha, float) is False:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if isinstance(step, int) is False:
                raise TypeError("step must be an integer")
            if not 0 < step <= iterations:
                raise ValueError("step must be positive and <= iterations")
            cost = self.evaluate(X, Y)[1]
            if verbose:
                print("Cost after", 0, "iterations:", cost)
            if graph:
                x = [0]
                y = [cost]
        iteration = 0
        next_step = 0
        while iteration < iterations:
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
            iteration += 1
            next_step += 1
            if next_step == step or iteration == iterations:
                next_step = 0
                if verbose or graph:
                    cost = self.evaluate(X, Y)[1]
                    if verbose:
                        print("Cost after", iteration, "iterations:", cost)
                    if graph:
                        x.append(iteration)
                        y.append(cost)
        if graph:
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.plot(x, y)
        return self.evaluate(X, Y)

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
