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
    (m, h, w), (h_k, w_k) = images.shape, kernel.shape

    h = h - h_k + 1
    w = w - w_k + 1
    convolved = np.zeros(shape=(m, h, w))

    for row in range(h):
        for col in range(w):
            part = images[:, row:(row + h_k), col:(col + w_k)]
            convolved[:, row, col] = np.sum(part * kernel, axis=(1, 2))
    return convolved
