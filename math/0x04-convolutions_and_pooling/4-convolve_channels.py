#!/usr/bin/env python3
"""
Provides a function to perform a convolution on images
"""
# pylint: disable=invalid-name
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images
    Arguments:
        images:
            a numpy.ndarray of shape (m, h, w, c) containing images, where
            m is the number of images,
            h is the height in pixels of each image,
            w is the width in pixels of each image,
            c is the number of channels in each image
        kernel:
            a numpy.ndarray of shape (hk, wk, c) containing the kernel, where
            hk is the height in pixels of the kernel,
            wk is the width in pixels of the kernel,
            c is the number of channels in each image
        padding:
            either ‘same’, ‘valid’, or a tuple of (hp, wp), where
            ‘same’ performs a same convolution,
            ‘valid’ performs a valid convolution,
            hp is the height of the padding for each image,
            wp is the width of the padding for each image
        stride:
            a tuple of (hs, ws), where
            hs is the height of the stride for each image,
            ws is the width of the stride for each image
    Return:
        a numpy.ndarray containing the convolved images
    """
    (m, h, w, _), (hk, wk, _), (hs, ws) = images.shape, kernel.shape, stride

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
            part = images[:, rows, cols, :]
            convolved[:, row, col, :] = np.sum(part * kernel, axis=(1, 2, 3))
    return convolved
