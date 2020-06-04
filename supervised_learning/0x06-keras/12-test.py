#!/usr/bin/env python3
"""
Provides a function to test a model
"""
# pylint: disable=invalid-name,too-many-arguments
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network
    Arguments:
        network: the model to test
        data: an array containing the input data
        labels: a one-hot encoded array containing the correct labels
        verbose: whether or not output should be printed during testing
    Return:
        the loss and accuracy of the model, respectively
    """
    return network.evaluate(x=data, y=labels, verbose=verbose)
