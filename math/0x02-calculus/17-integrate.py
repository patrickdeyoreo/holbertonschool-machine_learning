#!/usr/bin/env python3
"""Provides a function to calculate the integral of a polynomial"""


def poly_integral(poly, C=0):
    """Calculates the integral of a polynomial"""
    try:
        poly = poly[:]
        while poly[-1] == 0:
            poly.pop()
        poly = [coeff / power if coeff / power % 1 else coeff // power
                for power, coeff in enumerate(poly, 1)]
        poly.insert(0, C)
    except IndexError:
        poly = [C]
    except (TypeError, ValueError):
        poly = None
    return poly
