#!/usr/bin/env python3
"""
Provide a function to add two arrays element-wise.
"""

def add_arrays(arr1: list[int], arr2: list[int]) -> list[int]:
    """
    Add two arrays element-wise.
    """
    if len(arr1) != len(arr2):
        return None
    return [a + b for a, b in zip(arr1, arr2)]
