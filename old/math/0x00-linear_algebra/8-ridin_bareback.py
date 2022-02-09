#!/usr/bin/env python3
"""Provides a function to muliply two 2D matrices"""


def mat_mul(mat1, mat2):
    """Performs multiplication of two 2-dimensional matrices"""
    if len(mat1[0]) == len(mat2):
        return [[sum(a * b for a, b in zip(row, col)) for col in zip(*mat2)]
                for row in mat1]
    return None
