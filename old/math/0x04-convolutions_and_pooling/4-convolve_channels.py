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
        images: a np.ndarray of shape (m, h, w, c) of images, where
                m is the number of images,
                h is the height in pixels of each image,
                w is the width in pixels of each image,
                c is the number of channels in each image
        kernel: a np.ndarray of shape (hk, wk, c) as the kernel, where
                hk is the height in pixels of the kernel,
                wk is the width in pixels of the kernel,
                c is the number of channels in each image
        padding: either ‘same’, ‘valid’, or a tuple (hp, wp), where
                ‘same’ produces a same convolution,
                ‘valid’ produces a valid convolution,
                hp is the height in pixels of the padding,
                wp is the width in pixels of the padding
        stride: a tuple (hs, ws) as the stride, where
                hs is the height in pixels of the stride,
                ws is the width in pixels of the stride
    Return:
        a numpy.ndarray containing the convolved images
    """
    # pylint: disable=too-many-locals
    m, h, w, _ = images.shape
    hk, wk, _ = kernel.shape
    hs, ws = stride

    if isinstance(padding, tuple):
        hp, wp = padding
    elif padding == 'same':
        hp = ((hs - 1) * h - hs + hk + 1) // 2
        wp = ((ws - 1) * w - ws + wk + 1) // 2
    elif padding == 'valid':
        hp = wp = 0

    images = np.pad(images, ((0,), (hp,), (wp,), (0,)), mode='constant')
    h = (h - hk + 2 * hp) // hs + 1
    w = (w - wk + 2 * wp) // ws + 1
    conv = np.zeros(shape=(m, h, w))
    for row in range(h):
        for col in range(w):
            rows = slice(row * hs, row * hs + hk)
            cols = slice(col * ws, col * ws + wk)
            part = images[:, rows, cols] * kernel
            conv[:, row, col] = np.sum(part, axis=(1, 2, 3))
    return conv
