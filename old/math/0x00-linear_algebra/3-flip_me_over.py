#!/usr/bin/env python3
"""Provides a function to compute the transpose of a 2-dimensional matrix"""


def matrix_transpose(matrix):
    """Computes the transpose of a 2D matrix"""
    return list(map(list, zip(*matrix)))
