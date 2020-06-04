#!/usr/bin/env python3
"""
Provides a function to trains a keras model using mini-batch gradient descent
"""
# pylint: disable=invalid-name,too-many-arguments
import tensorflow.keras as K


def train_model(
        network, data, labels, batch_size, epochs,
        validation_data=None, early_stopping=False, patience=0,
        learning_rate_decay=False, alpha=0.1, decay_rate=1,
        save_best=False, filepath=None, verbose=True, shuffle=False):
    """
    Trains a keras model using mini-batch gradient descent
    Arguments:
        network: the model to train
        data: a numpy.ndarray of shape (m, nx) containing the input data
        labels: a one-hot numpy.ndarray of shape (m, classes) containing labels
        batch_size: the size of each batch of mini-batch gradient descent
        epochs: the number of times to pass through all the data
        validation_data: the data to validate the model with
        early_stopping: whether or not to end training upon diminishing returns
        patience: the number of insufficient passes to allow before stopping
        save_best: whether or not to save the best iteration of the model
        filepath: the path at which the model should be saved
        verbose: whether or not output should be printed during training
        shuffle: whether or not to shuffle batches every epoch
    Return:
        the History object produced by training the model
    """
    def scheduler(epoch):
        """Takes an epoch index and returns a learning rate"""
        return alpha / (1 + decay_rate * epoch)

    callbacks = []
    if validation_data is not None and early_stopping:
        callbacks.append(
            K.callbacks.EarlyStopping(monitor='val_loss', patience=patience))
    if validation_data is not None and learning_rate_decay:
        callbacks.append(
            K.callbacks.LearningRateScheduler(scheduler, verbose=1))
    if save_best:
        callbacks.append(
            K.callbacks.ModelCheckpoint(filepath, save_best_only=True))

    return network.fit(
        x=data, y=labels, batch_size=batch_size, epochs=epochs,
        callbacks=callbacks, validation_data=validation_data,
        verbose=verbose, shuffle=shuffle)
