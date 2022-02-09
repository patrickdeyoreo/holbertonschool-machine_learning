#!/usr/bin/env python3
"""Provides a function to calculate the integral of a polynomial"""


def poly_integral(poly, C=0):
    """Calculates the integral of a polynomial"""
    try:
        constant = float(C) if C % 1 else int(C)
        integral = poly[:]
        while integral[-1] == 0:
            integral.pop()
        integral = [coeff / power if coeff / power % 1 else coeff // power
                    for power, coeff in enumerate(integral, 1)]
        integral.insert(0, constant)
        return integral
    except IndexError:
        return None if len(poly) == 0 else [constant]
    except (TypeError, ValueError):
        return None
