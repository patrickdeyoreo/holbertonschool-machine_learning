#!/usr/bin/env python3
"""Trains a convolutional neural network to classify the CIFAR-10 dataset."""
# pylint: disable=invalid-name

from tensorflow import keras as K

MODEL = K.applications.inception_v3

