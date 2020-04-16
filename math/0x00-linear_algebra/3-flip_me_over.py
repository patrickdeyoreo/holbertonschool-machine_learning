#!/usr/bin/env python3
"""Provides a function to compute the transpose of a 2-dimensional matrix"""


def matrix_transpose(matrix):
    """Computes the transpose of a 2D matrix"""
    trans = [[None] * len(matrix) for _ in range(len(matrix[0]))]
    for row in range(len(trans)):
        for col in range(len(trans[0])):
            trans[row][col] = matrix[col][row]
    return trans
