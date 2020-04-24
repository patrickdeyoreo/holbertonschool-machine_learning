#!/usr/bin/env python3
"""Provides a function to calculate the derivative of a polynomial"""


def poly_derivative(poly):
    """Calculates the derivative of a polynomial"""
    try:
        _, *tail = poly
    except (TypeError, ValueError):
        return None
    if not tail or not any(tail):
        return [0]
    try:
        return [power * coeff for power, coeff in enumerate(tail, 1)]
    except (TypeError, ValueError):
        return None
