#!/usr/bin/env python3
"""
Provides a function to trains a keras model using mini-batch gradient descent
"""
# pylint: disable=invalid-name,too-many-arguments
import tensorflow.keras as K


def train_model(
        network, data, labels, batch_size, epochs,
        validation_data=None, verbose=True, shuffle=False):
    """
    Trains a keras model using mini-batch gradient descent
    Arguments:
        network: the model to train
        data: a numpy.ndarray of shape (m, nx) containing the input data
        labels: a one-hot numpy.ndarray of shape (m, classes) containing labels
        batch_size: the size of each batch of mini-batch gradient descent
        epochs: the number of times to pass through all the data
        validation_data: the data to validate the model with
        verbose: whether or not output should be printed during training
        shuffle: whether or not to shuffle batches every epoch
    Return:
        the History object produced by training the model
    """
    return network.fit(
        x=data, y=labels, batch_size=batch_size, epochs=epochs,
        validation_data=validation_data, verbose=verbose, shuffle=shuffle)
