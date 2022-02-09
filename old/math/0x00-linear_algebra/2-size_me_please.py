#!/usr/bin/env python3
"""Provides a function to compute the shape of a matrix"""


def matrix_shape(matrix):
    """Computes the shape of a matrix"""
    shape = [len(matrix)]
    if len(matrix) != 0 and isinstance(matrix[0], list):
        shape += matrix_shape(matrix[0])
    return shape
