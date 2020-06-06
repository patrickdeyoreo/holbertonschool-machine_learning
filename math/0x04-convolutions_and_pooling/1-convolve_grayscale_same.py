#!/usr/bin/env python3
"""
Provides a function to perform a same convolution on grayscale images
"""
# pylint: disable=invalid-name
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscales images
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

    h_p = int(h_k / 2)
    w_p = int(w_k / 2)
    images = np.pad(images, ((0,), (h_p,), (w_p,)), mode='constant')

    convolved = np.zeros(shape=(m, h, w))

    for row in range(h):
        for col in range(w):
            part = images[:, row:(row + h_k), col:(col + w_k)]
            convolved[:, row, col] = np.sum(part * kernel, axis=(1, 2))

    return convolved
