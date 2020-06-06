#!/usr/bin/env python3
"""
Provides a function to perform a convolution on grayscale images with padding
"""
# pylint: disable=invalid-name
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
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
        padding: either a tuple of (hp, wp), ‘same’, or ‘valid’, where
                 ‘same’ performs a same convolution,
                 ‘valid’ performs a valid convolution,
                 hp is the padding for the height of the image, and
                 wp is the padding for the width of the image
        stride: a tuple of (hs, ws), where
                hs is the stride for the height of the image
                ws is the stride for the width of the image
    Return:
        a np.ndarray containing the convolved images
    """
    (m, h, w), (hk, wk), (hs, ws) = images.shape, kernel.shape, stride

    if isinstance(padding, tuple):
        hp, wp = padding
    elif padding == 'same':
        hp = ((hs - 1) * h - hs + hk + 1) // 2
        wp = ((ws - 1) * w - ws + wk + 1) // 2
    elif padding == 'valid':
        hp = wp = 0

    images = np.pad(images, pad_width=((0,), (hp,), (wp,)), mode='constant')
    h = (h - hk + 2 * hp) // hs + 1
    w = (w - wk + 2 * wp) // ws + 1
    convolved = np.zeros(shape=(m, h, w))
    for row in range(h):
        for col in range(w):
            rows = slice(row * hs, row * hs + hk)
            cols = slice(col * ws, col * ws + wk)
            part = images[:, rows, cols]
            convolved[:, row, col] = np.sum(part * kernel, axis=(1, 2))
    return convolved
