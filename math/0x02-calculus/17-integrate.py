#!/usr/bin/env python3
"""Provides a function to calculate the integral of a polynomial"""


def poly_integral(poly, C=0):
    """Calculates the integral of a polynomial"""
    res = [C]
    try:
        res += [coeff / power if coeff / power % 1 else coeff // power
                for power, coeff in enumerate(poly, 1)]
    except (TypeError, ValueError):
        return None
    return res
