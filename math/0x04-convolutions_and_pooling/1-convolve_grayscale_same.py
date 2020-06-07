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
        images: a np.ndarray of shape (m, h, w) of images, where
                m is the number of images,
                h is the height in pixels of the images,
                w is the width in pixels of the images
        kernel: a np.ndarray with shape (hk, wk) as the kernel, where
                hk is the height in pixels of the kernel,
                wk is the width in pixels of the kernel
    Return:
        a np.ndarray containing the convolved images
    """
    (m, h, w), (hk, wk) = images.shape, kernel.shape

    hp = hk // 2
    wp = wk // 2
    images = np.pad(images, pad_width=((0,), (hp,), (wp,)), mode='constant')
    conv = np.zeros(shape=(m, h, w))
    for row in range(h):
        for col in range(w):
            rows = slice(row, row + hk)
            cols = slice(col, col + wk)
            part = images[:, rows, cols] * kernel
            conv[:, row, col] = np.sum(part, axis=(1, 2))
    return conv
