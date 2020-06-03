#!/usr/bin/env python3
"""
Provides a function to set up Adam optimization for a keras model with
categorical crossentropy loss and accuracy metrics
"""
# pylint: disable=invalid-name
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Sets up Adam optimization for a keras model with categorical crossentropy
    loss and accuracy metric
    Arguments:
        network: the model to optimize
        alpha: the learning rate
        beta1: the first Adam optimization parameter
        beta2: the second Adam optimization parameter
    """
    network.compile(
        optimizer=K.optimizers.Adam(lr=alpha, beta_1=beta1, beta_2=beta2),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
