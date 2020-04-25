#!/usr/bin/env python3
"""Provides a function to calculate the derivative of a polynomial"""


def poly_derivative(poly):
    """Calculates the derivative of a polynomial"""
    try:
        _, *poly = poly
        if any(poly):
            return [power * coeff for power, coeff in enumerate(poly, 1)]
    except (TypeError, ValueError):
        return None
    return [0]
