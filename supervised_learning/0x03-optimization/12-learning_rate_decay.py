#!/usr/bin/env python3
"""
Provides a function that creates a learning rate decay operation in
tensorflow using inverse time decay
"""
# pylint: disable=invalid-name,too-many-arguments
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Creates a learning rate decay operation using inverse time decay
    Arguments:
        alpha: the original learning rate
        decay_rate: the weight to determine the rate at which alpha will decay
        global_step: the number of passes of gradient descent that have elapsed
        decay_step: the number of passes of gradient descent that should occur
                    before alpha is decayed further
    Return:
        the learning rate decay operation
    """
    return tf.train.inverse_time_decay(
        learning_rate=alpha, global_step=global_step,
        decay_steps=decay_step, decay_rate=decay_rate, staircase=True)
