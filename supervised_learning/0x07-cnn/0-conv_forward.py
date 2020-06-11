#!/usr/bin/env python3
"""
Provides a function to perform forward propagation over a convolutional layer
of a neural network
"""
# pylint: disable=invalid-name
import numpy as np


def conv_forward(A_prev, W, b, activation, padding='same', stride=(1, 1)):
    """
    Performs convolutions on images
    Arguments:
        A_prev: np.ndarray of shape (m, h_prev, w_prev, c_prev) containing
            the output of the previous layer, where
            m is the number of examples,
            h_prev is the height of the previous layer,
            w_prev is the width of the previous layer,
            c_prev is the number of channels in the previous layer
        W: np.ndarray of shape (kh, kw, c_prev, c_new) containing kernels
            for the convolution, where
            kh is the filter height,
            kw is the filter width,
            c_prev is the number of channels in the previous layer,
            c_new is the number of channels in the output
        b: np.ndarray of shape (1, 1, 1, c_new) containing the biases to
            apply to the convolution
        activation: an activation function to apply to the convolution
        padding: either 'same' or 'valid', indicating the type of padding
        stride: a tuple (sh, sw) of the strides for the convolution, where
            sh is the stride for the height,
            sw is the stride for the width
    Return:
        the output of the convolutional layer
    """
    # pylint: disable=too-many-arguments,too-many-locals
    m, h, w, _ = A_prev.shape
    hk, wk, _, n = W.shape
    hs, ws = stride
    if padding == 'same':
        hp = ((hs - 1) * h - hs + hk + 1) // 2
        wp = ((ws - 1) * w - ws + wk + 1) // 2
    else:
        hp = wp = 0
    pad_width = ((0,), (hp,), (wp,), (0,))
    A_prev = np.pad(A_prev, pad_width, mode='constant')
    h = (h - hk + 2 * hp) // hs + 1
    w = (w - wk + 2 * wp) // ws + 1
    conv = np.zeros(shape=(m, h, w, n)) + b
    for kern in range(n):
        for row in range(h):
            for col in range(w):
                rows = slice(row * hs, row * hs + hk)
                cols = slice(col * ws, col * ws + wk)
                part = A_prev[:, rows, cols] * W[:, :, :, kern]
                conv[:, row, col, kern] += np.sum(part, axis=(1, 2, 3))
    return conv if activation is None else activation(conv)
