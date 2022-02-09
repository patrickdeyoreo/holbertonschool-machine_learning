#!/usr/bin/env python3
"""Provides a function to do back-propagation over a pooling layer."""
# pylint: disable=invalid-name
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Perform back-propagation over a pooling layer of a neural network.

    Arguments:
        dA: np.ndarray of shape (m, h_o, w_o, c_o) containing the partial
            derivatives with respect to the pooled output, where
            m is the number of examples,
            h_o is the height of the output,
            w_o is the width of the output,
            c_o is the number of channels in the output
        A_prev: np.ndarray of shape (m, h_i, w_i, c_i) containing input, where
            m is the number of examples,
            h_i is the height of the previous layer,
            w_i is the width of the previous layer,
            c_i is the number of channels in the previous layer
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
        the partial derivatives with respect to the previous layer (dX)
    """
    # pylint: disable=too-many-arguments,too-many-locals

    m, h_o, w_o, c_o = dA.shape
    h_k, w_k = kernel_shape
    h_s, w_s = stride

    if mode == 'max':
        def mask(A):
            """Mask for max pooling."""
            return np.where(A < np.max(A), 0, 1)
    else:
        def mask(A):
            """Mask for average pooling."""
            return np.ones_like(A) / (h_k * w_k)

    dX = np.zeros(A_prev.shape)

    for kern in range(c_o):
        for row in range(h_o):
            rows = slice(row * h_s, row * h_s + h_k)
            for col in range(w_o):
                cols = slice(col * w_s, col * w_s + w_k)
                for img in range(m):
                    X = mask(A_prev[img, rows, cols, kern])
                    dX[img, rows, cols, kern] += X * dA[img, row, col, kern]
    return dX
