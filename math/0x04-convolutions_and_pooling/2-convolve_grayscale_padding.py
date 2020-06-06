#!/usr/bin/env python3
"""
Provides a function to perform a convolution on grayscale images with padding
"""
# pylint: disable=invalid-name
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
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
        padding: a tuple of (h_p, w_p), where
                 h_p is the padding for the height of the image, and
                 w_p is the padding for the width of the image
    Return:
        a numpy.ndarray containing the convolved images
    """
    (m, h, w), (h_k, w_k), (h_p, w_p) = images.shape, kernel.shape, padding

    images = np.pad(images, pad_width=((0,), (h_p,), (w_p,)), mode='constant')
    h = h - h_k + 2 * h_p + 1
    w = w - w_k + 2 * w_p + 1
    convolved = np.zeros(shape=(m, h, w))

    for row in range(h):
        for col in range(w):
            part = images[:, row:(row + h_k), col:(col + w_k)]
            convolved[:, row, col] = np.sum(part * kernel, axis=(1, 2))
    return convolved
