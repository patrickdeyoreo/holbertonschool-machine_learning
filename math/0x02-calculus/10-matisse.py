#!/usr/bin/env python3
"""Provides a function to calculate the derivative of a polynomial"""


def poly_derivative(poly):
    """Calculates the derivative of a polynomial"""
    try:
        _, *tail = poly
        if tail and any(tail):
            return [power * coeff for power, coeff in enumerate(tail, 1)]
    except (TypeError, ValueError):
        return None
    return [0]
