#!/usr/bin/env python3
"""Provides a function to perform element-wise operations on a numpy.ndarray"""


def np_elementwise(mat1, mat2):
    """Performs addition, subtraction, multiplication and division"""
    return (mat1 + mat2), (mat1 - mat2), (mat1 * mat2), (mat1 / mat2)
