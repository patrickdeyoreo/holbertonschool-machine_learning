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
        images: a np.ndarray of shape (m, h, w, c) of images, where
                m is the number of images,
                h is the height in pixels of each image,
                w is the width in pixels of each image,
                c is the number of channels in each image
        kernel_shape: a tuple (hk, wk) as the kernel shape, where
                hk is the height in pixels of the kernel,
                wk is the width in pixels of the kernel,
        stride: a tuple of (hs, ws)  as the stride, where
                hs is the height in pixels of the stride,
                ws is the width in pixels of the stride
        mode: indicates the type of pooling, where
                'max' specifies max pooling,
                'avg' specifies average pooling
    Return:
        a np.ndarray containing the pooled images
    """
    # pylint: disable=too-many-locals
    m, h, w, c = images.shape
    hk, wk = kernel_shape
    hs, ws = stride
    h = (h - hk) // hs + 1
    w = (w - wk) // ws + 1
    npdict = vars(np)
    npdict['avg'] = np.mean
    poolfn = npdict.get(mode.lower(), mode)
    pooled = np.zeros(shape=(m, h, w, c))
    for row in range(h):
        for col in range(w):
            rows = slice(row * hs, row * hs + hk)
            cols = slice(col * ws, col * ws + wk)
            part = images[:, rows, cols]
            pooled[:, row, col] = poolfn(part, axis=(1, 2))
    return pooled
