#!/usr/bin/env python3
"""Provides a function to calculate the summation of i^2 from i=1 to n"""


def summation_i_squared(n):
    """Calculates the summation of i^2 from i=1 to n"""
    if isinstance(n, int):
        return sum(map(lambda i: i ** 2, range(1, n + 1)))
    return None
