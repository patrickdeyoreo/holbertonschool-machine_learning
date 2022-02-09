#!/usr/bin/env python3
"""
Provides functions to save and load models
"""
# pylint: disable=invalid-name
import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire model
    Arguments:
        network: the model to save
        filename: the path at which to save the model
    """
    return K.models.save_model(network, filename)


def load_model(filename):
    """
    Loads an entire model
    Arguments:
        filename: the path from which to load the model
    Return:
        the loaded model
    """
    return K.models.load_model(filename)
