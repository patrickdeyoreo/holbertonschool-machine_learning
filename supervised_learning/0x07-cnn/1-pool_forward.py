#!/usr/bin/env python3
"""
Provides a function to perform forward propagation over a pooling layer
of a neural network
"""
# pylint: disable=invalid-name
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward prop over a pooling layer of a neural network
    Arguments:
        A_prev: np.ndarray of shape (m, h_i, w_i, c_i) containing input, where
                m is the number of examples,
                h_i is the height of the previous layer,
                w_i is the width of the previous layer,
                c is the number of channels in the previous layer
        kernel_shape: a tuple (h_k, w_k) specifiying the kernel size, where
                h_k is the filter height,
                w_k is the filter width,
        stride: a tuple (h_s, w_s) specifying the stride size, where
                h_s is the height of the stride,
                w_s is the width of the stride
        max: either 'max' or 'avg', indicating the type of pooling
    Return:
        the output of the pooling layer
    """
    # pylint: disable=too-many-arguments,too-many-locals
    m, h_i, w_i, c = A_prev.shape
    h_k, w_k = kernel_shape
    h_s, w_s = stride
    h_o = (h_i - h_k) // h_s + 1
    w_o = (w_i - w_k) // w_s + 1
    poolfn = np.max if mode.lower() == 'max' else np.mean
    pooled = np.zeros(shape=(m, h_o, w_o, c))
    for i in range(h_o):
        rows = slice(i * h_s, i * h_s + h_k)
        for j in range(w_o):
            cols = slice(j * w_s, j * w_s + w_k)
            part = A_prev[:, rows, cols]
            pooled[:, i, j] = poolfn(part, axis=(1, 2))
    return pooled
