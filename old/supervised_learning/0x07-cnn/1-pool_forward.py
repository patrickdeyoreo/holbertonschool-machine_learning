#!/usr/bin/env python3
"""Provides a function to do forward propagation over a pooling layer."""
# pylint: disable=invalid-name
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Perform forward propagation over a pooling layer of a neural network.

    Arguments:
        A_prev: np.ndarray of shape (m, h_i, w_i, c_i) containing input, where
            m is the number of examples,
            h_i is the height of the previous layer,
            w_i is the width of the previous layer,
            c_o is the number of channels in the previous layer
        kernel_shape: a tuple (h_k, w_k) specifiying the kernel size, where
            h_k is the filter height,
            w_k is the filter width,
        stride: a tuple (h_s, w_s) specifying the stride size, where
            h_s is the height of the stride,
            w_s is the width of the stride
        mode: either 'max' or 'avg', indicating the type of pooling, where
            'max' specifies a max pooling,
            'avg' specifies a average pooling
    Returns:
        the output of the pooling layer
    """
    # pylint: disable=too-many-arguments,too-many-locals

    m, h_i, w_i, c_o = A_prev.shape
    h_k, w_k = kernel_shape
    h_s, w_s = stride

    h_o = (h_i - h_k) // h_s + 1
    w_o = (w_i - w_k) // w_s + 1

    f = np.max if mode == 'max' else np.mean
    Z = np.zeros(shape=(m, h_o, w_o, c_o))

    for row in range(h_o):
        rows = slice(row * h_s, row * h_s + h_k)
        for col in range(w_o):
            cols = slice(col * w_s, col * w_s + w_k)
            X = A_prev[:, rows, cols]
            Z[:, row, col] = f(X, axis=(1, 2))

    return Z
