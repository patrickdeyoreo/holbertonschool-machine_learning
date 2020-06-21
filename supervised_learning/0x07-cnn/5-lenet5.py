#!/usr/bin/env python3
"""Provides an implementation of the LeNet-5 neural network in Keras."""
# pylint: disable=invalid-name

import tensorflow.keras as K


def lenet5(X):
    """
    Build the LeNet-5 network.
    """
    layers = (
        K.layers.Conv2D(
            filters=6, kernel_size=5, padding='same', activation='relu',
            kernel_initializer=K.initializers.he_normal()
        ),
        K.layers.MaxPool2D(
            pool_size=2, strides=2
        ),
        K.layers.Conv2D(
            filters=16, kernel_size=5, padding='valid', activation='relu',
            kernel_initializer=K.initializers.he_normal()
        ),
        K.layers.MaxPool2D(
            pool_size=2, strides=2
        ),
        K.layers.Flatten(),
        K.layers.Dense(
            units=120, activation='relu',
            kernel_initializer=K.initializers.he_normal()
        ),
        K.layers.Dense(
            units=84, activation='relu',
            kernel_initializer=K.initializers.he_normal()
        ),
        K.layers.Dense(
            units=10, activation='softmax',
            kernel_initializer=K.initializers.he_normal()
        )
    )

    Z = X
    for layer in layers:
        Z = layer(Z)

    model = K.models.Model(inputs=X, outputs=Z)
    model.compile(
        optimizer=K.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
