#!/usr/bin/env python3
"""Provides a function to compute the shape of a matrix"""
from typing import List


def matrix_shape(matrix: List) -> List[int]:
    """Computes the shape of a matrix"""
    shape = [len(matrix)]
    if len(matrix) != 0 and isinstance(matrix[0], (list, tuple)):
        shape += matrix_shape(matrix[0])
    return shape
