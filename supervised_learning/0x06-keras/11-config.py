#!/usr/bin/env python3
"""
Provides functions to save and load model config
"""
# pylint: disable=invalid-name
import tensorflow.keras as K


def save_cofig(network, filename):
    """
    Saves the configuration of a model as JSON
    Arguments:
        network: the model of which to save the config
        filename: the path at which to save the config
    """
    config = network.to_json()
    with open(filename, 'w') as ostream:
        ostream.write(config)


def load_config(filename):
    """
    Creates a model from a JSON configuration
    Arguments:
        filename: the path from which to load the config
    Return:
        the loaded model
    """
    with open(filename, 'r') as istream:
        return K.models.model_from_json(istream.read())
