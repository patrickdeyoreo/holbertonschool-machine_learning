#!/usr/bin/env python3
"""
Provide a function to calculate the transpose of a matrix.
"""


def matrix_transpose(matrix: list) -> list:
    """
    Calculate the transpose of a matrix.
    """
    return list(map(list, zip(*matrix)))
