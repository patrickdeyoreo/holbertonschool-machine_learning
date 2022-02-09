#!/usr/bin/env python3
"""Provides a function to do forward propagation over a convolutional layer."""
# pylint: disable=invalid-name
import numpy as np


def conv_forward(A_prev, W, b, activation, padding='same', stride=(1, 1)):
    """
    Perform forward propagation over a convolutional layer of a neural network.

    Arguments:
        A_prev: np.ndarray of shape (m, h_i, w_i, c_i) containing input, where
            m is the number of examples,
            h_i is the height of the previous layer,
            w_i is the width of the previous layer,
            c_i is the number of channels in the previous layer
        W: np.ndarray of shape (h_k, w_k, c_i, c_o) containing kernels, where
            h_k is the filter height,
            w_k is the filter width,
            c_i is the number of channels in the previous layer,
            c_o is the number of channels in the output
        b: np.ndarray of shape (1, 1, 1, c_o) containing biases, where
            c_o is the number of channels in the output
        activation: an activation function to apply to the convolution, either
            a callable object or None
        padding: 'same' or 'valid', indicating the type of convolution, where
            'same' specifies a same convolution,
            'valid' specifies a valid convolution
        stride: a tuple (h_s, w_s) specifying the stride size, where
            h_s is the height of the stride,
            w_s is the width of the stride
    Returns:
        the output of the convolutional layer
    """
    # pylint: disable=too-many-arguments,too-many-locals

    m, h_i, w_i, _ = A_prev.shape
    h_k, w_k, _, c_o = W.shape
    h_s, w_s = stride

    if padding == 'same':
        h_p = ((h_s - 1) * h_i - h_s + h_k + 1) // 2
        w_p = ((w_s - 1) * w_i - w_s + w_k + 1) // 2
        pad_width = ((0, 0), (h_p, h_p), (w_p, w_p), (0, 0))
        A_prev = np.pad(A_prev, pad_width, mode='constant')
    else:
        h_p = w_p = 0

    h_o = (h_i - h_k + 2 * h_p) // h_s + 1
    w_o = (w_i - w_k + 2 * w_p) // w_s + 1

    Z = np.zeros(shape=(m, h_o, w_o, c_o)) + b

    for row in range(h_o):
        rows = slice(row * h_s, row * h_s + h_k)
        for col in range(w_o):
            cols = slice(col * w_s, col * w_s + w_k)
            for kern in range(c_o):
                K = W[..., kern]
                X = A_prev[:, rows, cols]
                Z[:, row, col, kern] += np.sum(K * X, axis=(1, 2, 3))

    return Z if activation is None else activation(Z)
