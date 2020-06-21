#!/usr/bin/env python3
"""Provides an implementation of the LeNet-5 neural network in TensorFlow."""
# pylint: disable=invalid-name

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    Calculate the accuracy of a model.

    Arguments:
        y: the correct labels
        y_pred: the predicted labels
    Returns:
        the accuracy of the prediction
    """
    true = tf.argmax(y, axis=1)
    pred = tf.argmax(y_pred, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(true, pred), tf.float32))


def calculate_loss(y, y_pred):
    """
    Calculate softmax cross-entropy loss.
    Arguments:
        y: the correct labels
        y_pred: the predicted labels
    Returns:
        the loss of the prediction
    """
    return tf.losses.softmax_cross_entropy(logits=y_pred, onehot_labels=y)


def create_train_op(loss):
    """
    Define an operation to perform gradient descent.
    Arguments:
        loss: the loss to be minimize
    Return:
        a training operation using Adam optimization
    """
    return tf.train.AdamOptimizer().minimize(loss)


def lenet5(x, y):
    """
    Build the LeNet-5 network.
    """
    initializer = tf.contrib.layers.variance_scaling_initializer()
    layers = [
        tf.layers.Conv2D(
            filters=6, kernel_size=5, padding='same',
            activation=tf.nn.relu,
            kernel_initializer=initializer
        ),
        tf.layers.MaxPooling2D(
            pool_size=2, strides=2
        ),
        tf.layers.Conv2D(
            filters=16, kernel_size=5, padding='valid',
            activation=tf.nn.relu,
            kernel_initializer=initializer
        ),
        tf.layers.MaxPooling2D(
            pool_size=2,
            strides=2
        ),
        tf.layers.Flatten(),
        tf.layers.Dense(
            units=120,
            activation=tf.nn.relu,
            kernel_initializer=initializer
        ),
        tf.layers.Dense(
            units=84,
            activation=tf.nn.relu,
            kernel_initializer=initializer
        ),
        tf.layers.Dense(
            units=10,
            kernel_initializer=initializer
        )
    ]

    z = x
    for layer in layers:
        z = layer(z)

    loss = calculate_loss(y, z)
    train_op = create_train_op(loss)
    y_pred = tf.nn.softmax(z)
    accuracy = calculate_accuracy(y, y_pred)
    return (y_pred, train_op, loss, accuracy)
