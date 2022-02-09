#!/usr/bin/env python3
"""
Provides a function that builds, trains, and saves a neural network model in
tensorflow using Adam optimization, mini-batch gradient descent, learning rate
decay, and batch normalization
"""
# pylint: disable=invalid-name,too-many-arguments
import numpy as np
import tensorflow as tf


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
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    return (x, y)


def create_layer(prev, n, activation, batch_norm=False, epsilon=1e-8):
    """
    Creates a layer for a neural network
    Arguments:
        prev: the tensor output of the previous layer
        n: the number of nodes in the layer to create
        activation: the activation function
        batch_norm: if true, create a batch-norm layer
        epsilon: variance epsilon for batch-norm layer
    Returns:
        the tensor output of the layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG')
    layer = tf.layers.Dense(
        units=n, activation=None, kernel_initializer=init, name='layer')
    tensor = layer(prev)
    if batch_norm:
        mean, variance = tf.nn.moments(layer(prev), axes=[0])
        beta = tf.Variable(tf.zeros((1, n)), trainable=True, name='beta')
        gamma = tf.Variable(tf.ones((1, n)), trainable=True, name='gamma')
        tensor = tf.nn.batch_normalization(
            layer(prev), mean=mean, variance=variance,
            offset=beta, scale=gamma, variance_epsilon=epsilon)
    return tensor if activation is None else activation(tensor)


def forward_prop(x, layers, activations, epsilon=1e-8):
    """
    Performs forward propagation in a neural network
    Arguments:
        x: the placeholder for input
        layers: a list of the number of nodes in each layer
        activations: a list of activation functions to use
        epsilon: variance epsilon for batch-norm layer
    Return:
        the prediction of the network in tensor form
    """
    # pylint: disable=dangerous-default-value
    y_pred = x
    for index, (layer, activation) in enumerate(zip(layers, activations), 1):
        batch_norm = index < len(layers)
        y_pred = create_layer(y_pred, layer, activation, batch_norm, epsilon)
    return y_pred


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction
    Arguments:
        y: the correct labels
        y_pred: the predicted labels
    Return:
        a tensor containing the accuracy of the prediction
    """
    true = tf.argmax(y, 1)
    pred = tf.argmax(y_pred, 1)
    return tf.reduce_mean(tf.cast(tf.equal(pred, true), tf.float32))


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


def shuffle_data(X, Y):
    """
    Shuffles the data in two matrices in the same way
    Arguments:
        X: the first np.ndarray of shape (m, nx) to shuffle, where
           m is the number of data points, and
           nx is the number of features
        Y: the second np.ndarray of shape (m, ny) to shuffle, where
           m is the same number of data points as in X, and
           ny is the number of features in Y
    Return:
        the shuffled matrices
    """
    perm = np.random.permutation(X.shape[0])
    return (X[perm], Y[perm])


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Creates a training operation for a neural network in tensorflow using
    the Adam optimization algorithm
    Arguments:
        loss: the loss of the network
        alpha: the learning rate
        beta1: the weight used for the first moment
        beta2: the weight used for the second moment
        epsilon: a small number to avoid division by zero
    Return:
        the Adam optimization operation
    """
    optimizer = tf.train.AdamOptimizer(
        learning_rate=alpha, beta1=beta1, beta2=beta2, epsilon=epsilon)
    return optimizer.minimize(loss)


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
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    global_step = tf.Variable(0, trainable=False, name='global_step')
    alpha = learning_rate_decay(alpha, decay_rate, global_step, 1)

    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layers, activations, epsilon=epsilon)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)
    params = {'x', 'y', 'y_pred', 'loss', 'accuracy', 'train_op'}

    for name in params:
        tf.add_to_collection(name, locals()[name])

    if X_train.shape[0] % batch_size == 0:
        batches = X_train.shape[0] // batch_size
    else:
        batches = X_train.shape[0] // batch_size + 1

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as session:

        session.run(init)

        for epoch in range(epochs + 1):

            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(
                session.run(
                    loss,
                    feed_dict={x: X_train, y: Y_train})
            ))
            print("\tTraining Accuracy: {}".format(
                session.run(
                    accuracy,
                    feed_dict={x: X_train, y: Y_train})
            ))
            print("\tValidation Cost: {}".format(
                session.run(
                    loss,
                    feed_dict={x: X_valid, y: Y_valid})
            ))
            print("\tValidation Accuracy: {}".format(
                session.run(
                    accuracy,
                    feed_dict={x: X_valid, y: Y_valid})
            ))

            if epoch < epochs:

                session.run(global_step.assign(epoch))
                session.run(alpha)

                X_perm, Y_perm = shuffle_data(X_train, Y_train)

                step = 0
                while step < batches:

                    X_bat = X_perm[step * batch_size:(step + 1) * batch_size]
                    Y_bat = Y_perm[step * batch_size:(step + 1) * batch_size]

                    session.run(train_op, feed_dict={x: X_bat, y: Y_bat})

                    step += 1
                    if step % 100 == 0:
                        print("\tStep {}:".format(step))
                        print("\t\tCost: {}".format(
                            session.run(
                                loss,
                                feed_dict={x: X_bat, y: Y_bat})
                        ))
                        print("\t\tAccuracy: {}".format(
                            session.run(
                                accuracy,
                                feed_dict={x: X_bat, y: Y_bat})
                        ))

        return saver.save(session, save_path)
