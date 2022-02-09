#!/usr/bin/env python3
"""Provides a class ``Exponential'' to represent an exponential distribution"""


class Exponential:
    """Represents an exponential distribution"""

    _e = 2.7182818285

    def __init__(self, data=None, lambtha=1.0):
        """
        Initializes a exponential distribution
        Arguments:
            data: the data to be used to estimate the distribution
            lambtha: expected number of occurences in a given time frame
        """
        self.lambtha = lambtha
        self.data = data

    @property
    def data(self):
        """
        Gets the data of a exponential distribution
        Return:
            data
        """
        return self.__data

    @data.setter
    def data(self, value):
        """
        Sets the data of a exponential distribution
        Arguments:
            value: a list containing at least two data points (or None)
        """
        if value is None:
            self.__data = None
        elif not isinstance(value, list):
            raise TypeError("data must be a list")
        elif len(value) < 2:
            raise ValueError("data must contain multiple values")
        else:
            self.__data = value[:]
            self.__lambtha = len(value) / sum(value)

    @property
    def lambtha(self):
        """
        Gets the lambtha of a exponential distribution
        Return:
            lambtha
        """
        return self.__lambtha

    @lambtha.setter
    def lambtha(self, value):
        """
        Sets the lambtha of a exponential distribution
        Arguments:
            value: a positive number
        """
        if getattr(self, 'data', None) is not None:
            raise ValueError("cannot change lambtha unless data is None")
        if value <= 0:
            raise ValueError("lambtha must be a positive value")
        self.__lambtha = float(value)

    def pdf(self, x):   # pylint: disable=invalid-name
        """
        Calculates the value of the PDF for a given time period
        Arguments:
            x: the time period
        Return:
            If x is out of range, return 0.
            Otherwise, return the PDF value for x.
        """
        if x < 0:
            return 0
        return self.lambtha * self._e ** (-self.lambtha * x)

    def cdf(self, x):   # pylint: disable=invalid-name
        """
        Calculates the value of the CDF for a given time period
        Arguments:
            x: the time period
        Return:
            If x is out of range, return 0.
            Otherwise, return the CDF value for x.
        """
        if x < 0:
            return 0
        return 1 - self._e ** (-self.lambtha * x)
