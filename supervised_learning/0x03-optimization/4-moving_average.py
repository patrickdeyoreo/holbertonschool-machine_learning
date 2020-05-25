#!/usr/bin/env python3
"""Provides a function that calculates a weighted moving average"""
# pylint: disable=invalid-name


def moving_average(data: list, beta: float) -> list:
    """
    Calculates the weighted moving average of a data set using bias-correction
    Arguments:
        data: the data of which to calculate the moving average
        beta: the weight used for the moving average
    Return:
        a list containing the moving averages of data
    """
    result = []
    temp = 0
    for const, value in enumerate(data, 1):
        temp = temp * beta + (1 - beta) * value
        result.append(temp / (1 - beta ** const))
    return result
