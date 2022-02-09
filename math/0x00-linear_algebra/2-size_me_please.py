#!/usr/bin/env python3
"""
Provide a function to calculate the shape of a matrix.
"""

def matrix_shape(matrix: list) -> list:
    """
    Caculate the shape of a matrix.
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
