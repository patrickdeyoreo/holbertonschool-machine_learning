#!/usr/bin/env python3
"""Provides a function to calculate the derivative of a polynomial"""


def poly_derivative(poly):
    """Calculates the derivative of a polynomial"""
    if isinstance(poly, list) and all(isinstance(item, int) for item in poly):
        return [power * coeff for power, coeff in enumerate(poly)][1:]
    return None
