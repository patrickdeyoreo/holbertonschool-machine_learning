#!/usr/bin/env python3
"""
Provides a function to perform a convolution on grayscale images with padding
"""
# pylint: disable=invalid-name
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscales images with custom padding
    Arguments:
        images: a np.ndarray of shape (m, h, w) of images, where
                m is the number of images,
                h is the height in pixels of the images,
                w is the width in pixels of the images
        kernel: a np.ndarray with shape (hk, wk) as the kernel, where
                hk is the height in pixels of the kernel,
                wk is the width in pixels of the kernel
        padding: a tuple of the form (hp, wp), where
                hp is the height in pixels of the padding,
                wp is the width in pixels of the padding
    Return:
        a np.ndarray containing the convolved images
    """
    # pylint: disable=too-many-locals
    (m, h, w), (hk, wk), (hp, wp) = images.shape, kernel.shape, padding

    images = np.pad(images, pad_width=((0,), (hp,), (wp,)), mode='constant')
    h = h - hk + 2 * hp + 1
    w = w - wk + 2 * wp + 1
    conv = np.zeros(shape=(m, h, w))
    for row in range(h):
        for col in range(w):
            rows = slice(row, row + hk)
            cols = slice(col, col + wk)
            part = images[:, rows, cols] * kernel
            conv[:, row, col] = np.sum(part, axis=(1, 2))
    return conv
