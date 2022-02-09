#!/usr/bin/env python3
"""Provides a class ``DeepNeuralNetwork'' for binary classification"""
# pylint: disable=invalid-name

import pickle
import numpy as np
import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    """Represents a deep neural network for binary classification"""

    ACTIVATIONS = {'sig', 'tanh'}

    def __init__(self, nx, layers, activation='sig'):
        """
        Initializes a binary classification neuron
        Arguments:
            nx: the number of input features
            layers: a list representing the number of nodes in each layer
            activation: the name of the activation function (sig or tanh)
        """
        if isinstance(nx, int) is False:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if isinstance(layers, list) is False or len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")

        if activation not in self.ACTIVATIONS:
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__activation = activation
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
    def activation(self):
        """
        Get the name of the activation function
        Return:
            the name of the activation function
        """
        return self.__activation

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

    def evaluate(self, X, Y):
        """
        Evaluates performance of the network with the given outputs and targets
        Arguments:
            X: numpy.ndarray with shape (nx, m) that contains the input, where
               nx is the number of input features to the neuron, and
               m is the number of examples
            Y: one-hot numpy.ndarray of shape (classes, m) of correct labels
        Return:
            the neuron’s prediction and the cost of the network, respectively
        """
        A = self.forward_prop(X)[0]
        P = np.where(A == np.max(A, axis=0), 1, 0)
        c = self.cost(Y, A)
        return (P, c)

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
        activation = getattr(self, self.__activation)
        A = self.__cache['A0'] = X
        W = self.weights['W1']
        b = self.weights['b1']
        for index in range(1, self.L):
            A = self.__cache['A{}'.format(index)] = activation(W @ A + b)
            W = self.weights['W{}'.format(index + 1)]
            b = self.weights['b{}'.format(index + 1)]
        A = self.__cache['A{}'.format(self.L)] = self.softmax(W @ A + b)
        return (A, self.cache)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates a pass of gradient descent and updates the weights and bias
        Arguments:
            Y: one-hot numpy.ndarray of shape (classes, m) of correct labels
            cache: dictionary containing the intermediary values of the network
            alpha: the learning rate
        """
        activation_prime = getattr(self, "{}_prime".format(self.__activation))
        m = Y.shape[1]
        dZ = cache['A{}'.format(self.L)] - Y
        for index in reversed(range(self.L)):
            A = cache['A{}'.format(index)]
            W = self.weights['W{}'.format(index + 1)]
            dW = dZ @ A.T / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            dZ = W.T @ dZ * activation_prime(A)
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
            Y: one-hot numpy.ndarray of shape (classes, m) of correct labels
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
            Y: one-hot numpy.ndarray of shape (classes, m) of correct labels
            A: numpy.ndarray with shape (classes, m) of the activated output
        Return:
            the cost
        """
        m = Y.shape[1]
        # log_probs = -np.log(A[Y.astype(int), range(m)])
        # return np.sum(log_probs) / m
        return -np.sum(Y * np.log(A)) / m

    @staticmethod
    def sig(X):
        """
        Sigmoid activation function
        Arguments:
            X: the x-values
        Return:
            the sigmoid of X
        """
        return 1 / (1 + np.exp(-X))

    @staticmethod
    def sig_prime(A):
        """
        Derivative of sigmoid activation function
        Arguments:
            A: the activation values
        Return:
            the derivative of sigmoid of A of X
        """
        return A * (1 - A)

    @staticmethod
    def tanh(X):
        """
        tanh activation function
        Arguments:
            X: the x-values
        Return:
            the tanh of X
        """
        exp_pos = np.exp(X)
        exp_neg = np.exp(-X)
        return (exp_pos - exp_neg) / (exp_pos + exp_neg)

    @staticmethod
    def tanh_prime(A):
        """
        Derivative of tanh activation function
        Arguments:
            A: the activation values
        Return:
            the derivative of tanh of A of X
        """
        return 1 - np.power(A, 2)

    @staticmethod
    def softmax(X):
        """
        Softmax activation function
        Arguments:
            X: the x-values
        Return:
            the softmax of X
        """
        # logits = np.exp(X)
        logits = np.exp(X - np.max(X))
        # return logits / np.sum(logits)
        return logits / np.sum(logits, axis=0, keepdims=True)
