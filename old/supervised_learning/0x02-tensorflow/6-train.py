#!/usr/bin/env python3
"""Provides a function to build, train, and save a neural network classifier"""
# pylint: disable=invalid-name

import tensorflow as tf

create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier
    Arguments:
        X_train: a numpy.ndarray containing the training input data
        Y_train: a numpy.ndarray containing the training labels
        X_valid: a numpy.ndarray containing the validation input data
        Y_valid: a numpy.ndarray containing the validation labels
        layer_sizes: a list of the number of nodes in each layer
        activations: a list of the activation functions for each layer
        alpha: the learning rate
        iterations: the number of iterations to train over
        save_path: the name of the file in which to save the model
    Return:
        the path to the saved model
    """
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    accuracy = calculate_accuracy(y, y_pred)
    loss = calculate_loss(y, y_pred)
    train_op = create_train_op(loss, alpha)
    for key in ('x', 'y', 'y_pred', 'accuracy', 'train_op'):
        tf.add_to_collection(key, locals().get(key))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as s:
        s.run(init)
        for i in range(iterations + 1):
            s.run(y_pred, feed_dict={x: X_train, y: Y_train})
            if i % 100 == 0 or i == iterations:
                print("After", i, "iterations:")
                print("\tTraining Cost:",
                      s.run(loss, feed_dict={x: X_train, y: Y_train}))
                print("\tTraining Accuracy:",
                      s.run(accuracy, feed_dict={x: X_train, y: Y_train}))
                print("\tValidation Cost:",
                      s.run(loss, feed_dict={x: X_valid, y: Y_valid}))
                print("\tValidation Accuracy:",
                      s.run(accuracy, feed_dict={x: X_valid, y: Y_valid}))
            if i != iterations:
                s.run(train_op, feed_dict={x: X_train, y: Y_train})
        saved = saver.save(s, save_path)
    return saved
