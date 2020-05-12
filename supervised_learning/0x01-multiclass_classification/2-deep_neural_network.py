#!/usr/bin/env python3
"""Provides a class ``DeepNeuralNetwork'' for binary classification"""
# pylint: disable=invalid-name

import pickle
import numpy as np
import matplotlib.pyplot as plt


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

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the network using a sigmoid
        activation function and updates the cache
        Arguments:
            X: numpy.ndarray with shape (nx, m) that contains the input, where
               nx is the number of input features to the neuron, and
               m is the number of examples
        Return:
            the activation state
        """
        self.__cache['A0'] = X
        for index in range(1, self.L + 1):
            W = self.weights['W{}'.format(index)]
            b = self.weights['b{}'.format(index)]
            X = self.__cache['A{}'.format(index)] = self.sigmoid(W @ X + b)
        return (X, self.cache)

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
        A = self.forward_prop(X)[0]
        P = np.where(A < 0.5, 0, 1)
        c = self.cost(Y, A)
        return (P, c)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates a pass of gradient descent and updates the weights and bias
        Arguments:
            Y: numpy.ndarray with shape (1, m) that contains the correct labels
            cache: dictionary containing the intermediary values of the network
            alpha: the learning rate
        """
        m = Y.shape[1]
        dZ = cache['A{}'.format(self.L)] - Y
        for index in reversed(range(self.L)):
            A = cache['A{}'.format(index)]
            W = self.weights['W{}'.format(index + 1)]
            dW = (1 / m) * (dZ @ A.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            dZ = (W.T @ dZ) * (A * (1 - A))
            self.__weights['W{}'.format(index + 1)] -= alpha * dW
            self.__weights['b{}'.format(index + 1)] -= alpha * db

    # pylint: disable=too-many-arguments,too-many-branches
    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the network
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
            self.forward_prop(X)
            self.gradient_descent(Y, self.cache, alpha)
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

    def save(self, filename):
        """
        Saves a pickled network
        Arguments:
            filename: the file to which the object should be saved (.pkl)
        """
        if not filename.endswith('.pkl'):
            filename = '.'.join((filename, 'pkl'))
        with open(filename, 'wb') as ostream:
            pickle.dump(self, ostream)

    @staticmethod
    def load(filename):
        """
        Loads a pickled network
        Arguments:
            filename: the file to which the object should be saved (.pkl)
        Return:
            the loaded object, or None if filename doesn’t exist
        """
        try:
            with open(filename, 'rb') as istream:
                return pickle.load(istream)
        except FileNotFoundError:
            return None

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
