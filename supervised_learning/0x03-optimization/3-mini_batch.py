#!/usr/bin/env python3
"""
Provides a function to train a neural network using mini-batch gradient descent
"""
# pylint: disable=invalid-name
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
    Trains neural network using mini-batch gradient descent
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
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(session, load_path)
        var = {
            'x': None,
            'y': None,
            'accuracy': None,
            'loss': None,
            'train_op': None,
            'y_pred': None
        }
        for name in var:
            var[name] = tf.get_collection(name)[0]

        epoch = 0
        for epoch in range(epochs + 1):
            loss_t = session.run(
                var['loss'],
                feed_dict={var['x']: X_train, var['y']: Y_train})
            accuracy_t = session.run(
                var['accuracy'],
                feed_dict={var['x']: X_train, var['y']: Y_train})
            loss_v = session.run(
                var['loss'],
                feed_dict={var['x']: X_valid, var['y']: Y_valid})
            accuracy_v = session.run(
                var['accuracy'],
                feed_dict={var['x']: X_valid, var['y']: Y_valid})
            print("After", epoch, "epochs:")
            print("\tTraining Cost:", loss_t)
            print("\tTraining Accuracy:", accuracy_t)
            print("\tValidation Cost:", loss_v)
            print("\tValidation Accuracy:", accuracy_v)

            if epoch < epochs:
                X_perm, Y_perm = shuffle_data(X_train, Y_train)
                batches, remainder = divmod(X_train.shape[0], batch_size)

                descent_step = 0
                for i in range(0, batches + 1):
                    descent_step += 1

                    # Important: make copies of X_perm and Y_perm
                    if i == batches:
                        if remainder != 0:
                            X_batch = X_perm[i * batch_size:]
                            Y_batch = Y_perm[i * batch_size:]
                        else:
                            break
                    else:
                        X_batch = X_perm[i * batch_size:(i + 1) * batch_size]
                        Y_batch = Y_perm[i * batch_size:(i + 1) * batch_size]

                    session.run(
                        var['train_op'],
                        feed_dict={var['x']: X_batch, var['y']: Y_batch})

                    if descent_step % 100 == 0:
                        loss_b = session.run(
                            var['loss'],
                            feed_dict={var['x']: X_batch, var['y']: Y_batch})
                        accuracy_b = session.run(
                            var['accuracy'],
                            feed_dict={var['x']: X_batch, var['y']: Y_batch})

                        print("\tStep", descent_step, end=":\n")
                        print("\t\tCost:", loss_b)
                        print("\t\tAccuracy:", accuracy_b)

        save_path = saver.save(session, save_path)
    return save_path
