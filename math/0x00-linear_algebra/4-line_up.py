#!/usr/bin/env python3
"""Provides a function to perform element-wise additon of two arrays"""


def add_arrays(arr1, arr2):
    """Performs element-wise additon of two arrays"""
    if len(arr1) == len(arr2):
        return [a + b for a, b in zip(arr1, arr2)]
    return None
