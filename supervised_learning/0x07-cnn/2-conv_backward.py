#!/usr/bin/env python3
"""
Provides a function to perform back propagation over a convolutional layer
of a neural network
"""
# pylint: disable=invalid-name
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding='same', stride=(1, 1)):
    """
    Performs back prop over a convolutional layer of a neural network
    Arguments:
        dZ: np.ndarray of shape (m, h, w, c) containing the partial
                derivatives with respect to the unactivated output, where
                m is the number of examples,
                h is the height of the output,
                w is the width of the output,
                c is the number of channels in the output
        A_prev: np.ndarray of shape (m, h_i, w_i, c_i) containing input, where
                m is the number of examples,
                h_i is the height of the previous layer,
                w_i is the width of the previous layer,
                c_i is the number of channels in the previous layer
        W: np.ndarray of shape (h_k, w_k, c_i, c) containing kernels, where
                h_k is the filter height,
                w_k is the filter width,
                c_i is the number of channels in the previous layer,
                c is the number of channels in the output
        b: np.ndarray of shape (1, 1, 1, c) containing biases, where
                c is the number of channels in the output
        padding: 'same' or 'valid', indicating the type of convolution, where
                'same' specifies a same convolution,
                'valid' specifies a valid convolution
        stride: a tuple (h_s, w_s) specifying the stride size, where
                h_s is the height of the stride,
                w_s is the width of the stride
    Return:
        the partial derivatives with respect to the previous layer (dX),
        to the kernels (dK), and to the biases (db), respectively
    """
    # pylint: disable=too-many-arguments,too-many-locals

    _, h, w, c = dZ.shape
    _, h_i, w_i, _ = A_prev.shape
    h_k, w_k, _, _ = W.shape
    h_s, w_s = stride

    if padding == 'same':
        h_p = ((h_s - 1) * h_i - h_s + h_k + 1) // 2
        w_p = ((w_s - 1) * w_i - w_s + w_k + 1) // 2
    else:
        h_p = w_p = 0

    dX = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    padding = ((0,), (h_p,), (w_p,), (0,))
    A_prev = np.pad(A_prev, padding, mode='constant')

    for row in range(h):
        rows = slice(row * h_s, row * h_s + h_k)
        for col in range(w):
            cols = slice(col * w_s, col * w_s + w_k)
            A = A_prev[:, rows, cols]
            for kern in range(c):
                X = dZ[:, row, col, kern].reshape((-1, 1, 1, 1))
                dX[:, rows, cols] += X * W[np.newaxis, ..., kern]
                dW[:, :, :, kern] += np.sum(X * A, axis=0)

    return (dX, dW, db)
