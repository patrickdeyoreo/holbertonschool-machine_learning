#!/usr/bin/env python3
"""
Provides a function to perform a valid convolution on grayscale images
"""
# pylint: disable=invalid-name
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscales images
    Arguments:
        images: a np.ndarray of shape (m, h, w) of grayscale images, where
                m is the number of images,
                h is the height in pixels of the images, and
                w is the width in pixels of the images
        kernel: a np.ndarray with shape (h_k, w_k) containing the kernel, where
                h_k is the height of the kernel, and
                w_k is the width of the kernel
    Return:
        a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    h_k, w_k = kernel.shape
    h_c = h - h_k + 1
    w_c = w - w_k + 1
    convolution = np.zeros(shape=(m, h_c, w_c), dtype=int)

    for row in range(h_c):
        for col in range(w_c):
            part = images[:, row:(row + h_k), col:(col + w_k)]
            convolution[:, row, col] = np.sum(part * kernel, axis=(1, 2))

    return convolution
