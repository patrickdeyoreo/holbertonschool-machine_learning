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
        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
