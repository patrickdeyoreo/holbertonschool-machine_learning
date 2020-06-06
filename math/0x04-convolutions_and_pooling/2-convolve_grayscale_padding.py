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
        kernel: a np.ndarray with shape (hk, wk) containing the kernel, where
                hk is the height of the kernel, and
                wk is the width of the kernel
        padding: a tuple of (hp, wp), where
                 hp is the padding for the height of the image, and
                 wp is the padding for the width of the image
    Return:
        a numpy.ndarray containing the convolved images
    """
    (m, h, w), (hk, wk), (hp, wp) = images.shape, kernel.shape, padding

    images = np.pad(images, pad_width=((0,), (hp,), (wp,)), mode='constant')
    h = h - hk + 2 * hp + 1
    w = w - wk + 2 * wp + 1
    convolved = np.zeros(shape=(m, h, w))
    for row in range(h):
        for col in range(w):
            part = images[:, row:(row + hk), col:(col + wk)]
            convolved[:, row, col] = np.sum(part * kernel, axis=(1, 2))
    return convolved
