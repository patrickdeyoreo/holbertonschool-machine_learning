#!/usr/bin/env python3
"""
Provides a function to perform convolutions on images
"""
# pylint: disable=invalid-name
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs convolutions on images
    Arguments:
        images:
            a np.ndarray of shape (m, h, w, c) containing images, where
            m is the number of images,
            h is the height in pixels of each image,
            w is the width in pixels of each image,
            c is the number of channels in each image
        kernel:
            a np.ndarray of shape (hk, wk, c, n) containing the kernels, where
            hk is the height in pixels of each kernel,
            wk is the width in pixels of each kernel,
            c is the number of channels in each image,
            n is the number of kernels
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
        a np.ndarray containing the convolved images
    """
    m, h, w, _ = images.shape
    hk, wk, _, n = kernels.shape
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
    convolved = np.zeros(shape=(m, h, w, n))
    for kern in range(n):
        for row in range(h):
            for col in range(w):
                rows = slice(row * hs, row * hs + hk)
                cols = slice(col * ws, col * ws + wk)
                part = images[:, rows, cols] * kernels[:, :, :, kern]
                convolved[:, row, col, kern] = np.sum(part, axis=(1, 2, 3))
    return convolved
