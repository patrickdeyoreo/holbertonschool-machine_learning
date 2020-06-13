#!/usr/bin/env python3
"""
Provides a function to perform back propagation over a convolutional layer
of a neural network
"""
# pylint: disable=invalid-name
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back prop over a convolutional layer of a neural network
    Arguments:
        dA: np.ndarray of shape (m, h, w, c) containing the partial
                derivatives with respect to the pooled output, where
                m is the number of examples,
                h is the height of the output,
                w is the width of the output,
                c is the number of channels in the output
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
    Return:
        the partial derivatives with respect to the previous layer (dA_prev)
    """
    # pylint: disable=too-many-arguments,too-many-locals

    _, h, w, _ = dA.shape
    h_k, w_k = kernel_shape
    h_s, w_s = stride

    f = np.max if mode == 'max' else np.mean
    dA_prev = np.zeros(A_prev.shape)

    for row in range(h):
        rows = slice(row * h_s, row * h_s + h_k)
        for col in range(w):
            cols = slice(col * w_s, col * w_s + w_k)
            A = A_prev[:, rows, cols]
            X = dA[:, row:row + 1, col:col + 1]
            Z = f(A, axis=(1, 2), keepdims=True)
            dA_prev[:, rows, cols] += f(A + X, axis=(1, 2), keepdims=True) - Z

    return dA_prev
