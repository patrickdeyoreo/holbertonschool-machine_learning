#!/usr/bin/env python3
"""
Provides a function to perform pooling on images
"""
# pylint: disable=invalid-name
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images
    Arguments:
        images: a np.ndarray of shape (m, h, w, c) containing images, where
            m is the number of images,
            h is the height in pixels of each image,
            w is the width in pixels of each image,
            c is the number of channels in each image
        kernel_shape: a tuple of (hk, wk) containing the kernel shape, where
            hk is the height in pixels of the kernel,
            wk is the width in pixels of the kernel,
        stride: a tuple of (hs, ws), where
            hs is the height of the stride for each image,
            ws is the width of the stride for each image
        mode: indicates the type of pooling, where
            'max' indicates max pooling,
            'avg' indicates average pooling
    Return:
        a np.ndarray containing the pooled images
    """
    # pylint: disable=too-many-locals
    m, h, w, c = images.shape
    hk, wk = kernel_shape
    hs, ws = stride
    h = (h - hk) // hs + 1
    w = (w - wk) // ws + 1
    pooled = np.zeros(shape=(m, h, w, c))
    pool_fn = {'avg': np.mean, 'max': np.max}.get(mode.lower(), mode)
    for row in range(h):
        for col in range(w):
            rows = slice(row * hs, row * hs + hk)
            cols = slice(col * ws, col * ws + wk)
            pooled[:, row, col] = pool_fn(images[:, rows, cols], axis=(1, 2))
    return pooled
