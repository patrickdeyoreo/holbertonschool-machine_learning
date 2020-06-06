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
        kernel: a np.ndarray with shape (h_k, w_k) containing the kernel, where
                h_k is the height of the kernel, and
                w_k is the width of the kernel
        padding: either a tuple of (h_p, w_p), ‘same’, or ‘valid’, where
                 ‘same’ performs a same convolution,
                 ‘valid’ performs a valid convolution,
                 h_p is the padding for the height of the image, and
                 w_p is the padding for the width of the image
        stride: a tuple of (h_s, w_s), where
                h_s is the stride for the height of the image
                w_s is the stride for the width of the image
    Return:
        a np.ndarray containing the convolved images
    """
    m, h, w = images.shape
    h_k, w_k = kernel.shape
    h_s, w_s = stride

    if isinstance(padding, tuple):
        h_p, w_p = padding
        images = np.pad(images, ((0,), (h_p,), (w_p,)), mode='constant')
    elif padding == 'same':
        h_p = int(np.ceil((h_s * (h - 1) + h_k - h) / 2))
        w_p = int(np.ceil((w_s * (w - 1) + w_k - w) / 2))
        images = np.pad(images, ((0,), (h_p,), (w_p,)), mode='constant')
    elif padding == 'valid':
        h_p = w_p = 0

    h_c = int((h - h_k + 2 * h_p) / h_s + 1)
    w_c = int((w - w_k + 2 * w_p) / w_s + 1)
    convolved = np.zeros(shape=(m, h_c, w_c))

    for row in range(h_c):
        for col in range(w_c):
            part = images[:, row*h_s:row*h_s+h_k, col*w_s:col*w_s+w_k]
            convolved[:, row, col] = np.sum(part*kernel, axis=(1, 2))

    return convolved
