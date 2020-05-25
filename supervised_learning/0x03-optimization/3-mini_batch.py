#!/usr/bin/env python3
"""
Provides a function to train a neural network using mini-batch gradient descent
"""
# pylint: disable=invalid-name,too-many-arguments,too-many-locals
import tensorflow as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(
        X_train, Y_train,
        X_valid, Y_valid,
        batch_size=32, epochs=5,
        load_path="/tmp/model.ckpt",
        save_path="/tmp/model.ckpt",
):
    """
    Trains a saved neural network using mini-batch gradient descent
    Arguments:
        X_train: np.ndarray of shape (m, 784) containing the training data
        Y_train: one-hot np.ndarray of shape (m, 10) of the training labels
        X_valid: np.ndarray of shape (m, 784) containing the validation data
        Y_valid: one-hot np.ndarray of shape (m, 10) of the validation labels
        batch_size: the number of data points in a batch
        epochs: the number of times to train through the whole dataset
        load_path: the path from which to load the model
        save_path: the path at which to save the model after training
    Return:
        the path at which the model was saved
    """
    with tf.Session() as session:

        saver = tf.train.import_meta_graph('.'.join((load_path, 'meta')))
        saver.restore(session, load_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        if X_train.shape[0] % batch_size == 0:
            batches = X_train.shape[0] // batch_size
        else:
            batches = X_train.shape[0] // batch_size + 1

        for epoch in range(epochs + 1):

            loss_t = session.run(
                loss,
                feed_dict={x: X_train, y: Y_train})
            accuracy_t = session.run(
                accuracy,
                feed_dict={x: X_train, y: Y_train})
            loss_v = session.run(
                loss,
                feed_dict={x: X_valid, y: Y_valid})
            accuracy_v = session.run(
                accuracy,
                feed_dict={x: X_valid, y: Y_valid})

            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(loss_t))
            print("\tTraining Accuracy: {}".format(accuracy_t))
            print("\tValidation Cost: {}".format(loss_v))
            print("\tValidation Accuracy: {}".format(accuracy_v))

            if epoch < epochs:

                X_perm, Y_perm = shuffle_data(X_train, Y_train)

                bat = 0
                while bat < batches:

                    X_bat = X_perm[bat * batch_size:(bat + 1) * batch_size]
                    Y_bat = Y_perm[bat * batch_size:(bat + 1) * batch_size]

                    session.run(
                        train_op,
                        feed_dict={x: X_bat, y: Y_bat})

                    bat += 1

                    if bat % 100 == 0:

                        loss_b = session.run(
                            loss,
                            feed_dict={x: X_bat, y: Y_bat})
                        accuracy_b = session.run(
                            accuracy,
                            feed_dict={x: X_bat, y: Y_bat})

                        print("\tStep {}:".format(bat))
                        print("\t\tCost: {}".format(loss_b))
                        print("\t\tAccuracy: {}".format(accuracy_b))

        return saver.save(session, save_path)
