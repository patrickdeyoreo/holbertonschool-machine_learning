#!/usr/bin/env python3
"""
Provides a function to make predictions
"""
# pylint: disable=invalid-name,too-many-arguments
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes predictions
    Arguments:
        network: the model to test
        data: an array containing the input data
        verbose: whether or not output should be printed during prediction
    Return:
        predictions for the data
    """
    return network.predict(x=data, verbose=verbose)
