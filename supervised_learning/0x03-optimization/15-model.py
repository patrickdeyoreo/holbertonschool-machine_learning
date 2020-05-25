#!/usr/bin/env python3
"""
Provides a function that builds, trains, and saves a neural network model in
tensorflow using Adam optimization, mini-batch gradient descent, learning rate
decay, and batch normalization
"""
# pylint: disable=invalid-name,too-many-arguments
import tensorflow as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data
train_mini_batch = __import__('3-mini_batch').train_mini_batch
create_Adam_op = __import__('10-Adam').create_Adam_op
learning_rate_decay = __import__('12-learning_rate_decay').learning_rate_decay
create_batch_norm_layer = __import__('14-batch_norm').create_batch_norm_layer


def create_placeholders(nx, classes):
    """
    Creates two placeholders for a neural network
    Arguments:
        nx: the number of feature columns in our data
        classes: the number of classes in our classifier
    Returns:
        x: the placeholder for the input data to the neural network
        y: the placeholder for the one-hot labels for the input data
    """
    return (tf.placeholder(tf.float32, shape=(None, nx), name='x'),
            tf.placeholder(tf.float32, shape=(None, classes), name='y'))


def create_layer(prev, n, activation):
    """
    Creates a layer for a neural network
    Arguments:
        prev: the tensor output of the previous layer
        n: the number of nodes in the layer to create
        activation: the activation function
    Returns:
        the tensor output of the layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG')
    layer = tf.layers.Dense(
        units=n, activation=activation, kernel_initializer=init, name='layer')
    return layer(prev)


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Performs forward propagation in a neural network
    Arguments:
        x: the placeholder for input
        layer_sizes: a list of the number of nodes in each layer
        activations: a list of activation functions to use
    Return:
        the prediction of the network in tensor form
    """
    # pylint: disable=dangerous-default-value
    y_pred = x
    for size, activation in zip(layer_sizes, activations):
        y_pred = create_layer(y_pred, size, activation)
    return y_pred


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction
    Arguments:
        y: a placeholder for the labels of the input
        y_pred: the predictions
    Return:
        a tensor containing the accuracy of the prediction
    """
    return tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1)), tf.float32))


def calculate_loss(y, y_pred):
    """
    Calculates softmax loss
    Arguments:
        y: the correct labels
        y_pred: the predicted labels
    Return:
        a tensor containing the loss of the prediction
    """
    return tf.losses.softmax_cross_entropy(logits=y_pred, onehot_labels=y)


def model(
        Data_train, Data_valid, layers, activations,
        alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
        decay_rate=1, batch_size=32, epochs=5,
        save_path='/tmp/model.ckpt'
):
    """
    Creates a batch normalization layer for a neural network in tensorflow
    Arguments:
        Data_train: a tuple containing the training inputs and labels
        Data_valid: a tuple containing the validation inputs and labels
        layers: a list containing the number of nodes in each layer
        activation: a list containing the activation functions for each layer
        alpha: the learning rate
        beta1: the weight for the first moment of Adam Optimization
        beta2: the weight for the second moment of Adam Optimization
        epsilon: a small number used to avoid division by zero
        decay_rate: the decay rate for inverse time decay of the learning rate
        batch_size: the number of data points that should be in a mini-batch
        epochs: the number of times to train through the whole dataset
        save_path: the path at which the model should be saved
    Return:
        the path at which the model was saved
    """
