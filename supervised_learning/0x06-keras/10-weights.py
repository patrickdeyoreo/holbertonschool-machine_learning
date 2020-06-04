#!/usr/bin/env python3
"""
Provides a function to save and load weights of models
"""
# pylint: disable=invalid-name
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """
    Saves the weights of a model
    Arguments:
        network: the model of which to save weights
        filename: the path at which to save weights
        save_format: the desired file format
    """
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """
    Loads the weights of a model
    Arguments:
        network: the model for which to load weights
        filename: the path from which to load weights
    """
    network.load_weights(filename)
