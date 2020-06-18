#!/usr/bin/env python3
"""Provides a function to do back-propagation over a convolutional layer."""
# pylint: disable=invalid-name

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding='same', stride=(1, 1)):
    """
    Perform back-propagation over a convolutional layer of a neural network.

    Arguments:
        dZ: np.ndarray of shape (m, h_o, w_o, c_o) containing the partial
            derivatives with respect to the unactivated output, where
            m is the number of examples,
            h_o is the height of the output,
            w_o is the width of the output,
            c_o is the number of channels in the output
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
        padding: 'same' or 'valid', indicating the type of convolution, where
            'same' specifies a same convolution,
            'valid' specifies a valid convolution
        stride: a tuple (h_s, w_s) specifying the stride size, where
            h_s is the height of the stride,
            w_s is the width of the stride
    Returns:
        the partial derivatives with respect to the previous layer (dX),
        to the kernels (dW), and to the biases (db), respectively
    """
    # pylint: disable=too-many-arguments,too-many-locals,unused-argument

    m, h_o, w_o, c_o = dZ.shape
    _, h_i, w_i, _ = A_prev.shape
    h_k, w_k, _, _ = W.shape
    h_s, w_s = stride

    if padding == 'same':
        h_p = ((h_s - 1) * h_i - h_s + h_k + 1) // 2
        w_p = ((w_s - 1) * w_i - w_s + w_k + 1) // 2
        pad_width = ((0, 0), (h_p, h_p), (w_p, w_p), (0, 0))
        A_prev = np.pad(A_prev, pad_width, mode='constant')

    dX = np.zeros(shape=A_prev.shape)
    dW = np.zeros(shape=W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for kern in range(c_o):
        K = W[..., kern]
        for row in range(h_o):
            rows = slice(row * h_s, row * h_s + h_k)
            for col in range(w_o):
                cols = slice(col * w_s, col * w_s + w_k)
                for img in range(m):
                    A = A_prev[img, rows, cols]
                    X = dZ[img, row, col, kern]
                    dX[img, rows, cols] += X * K
                    dW[..., kern] += A * X

    if padding == 'same':
        rows = slice(h_p, dX.shape[1] - h_p)
        cols = slice(w_p, dX.shape[2] - w_p)
        dX = dX[:, rows, cols]

    return (dX, dW, db)
