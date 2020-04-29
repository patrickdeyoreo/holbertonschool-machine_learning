#!/usr/bin/env python3
"""Provides a class ``Poisson'' to represent a poisson distribution"""


class Poisson:
    """Represents a poisson distribution"""

    _e = 2.7182818285

    def __init__(self, data=None, lambtha=1.0):
        """
        Initializes a poisson distribution
        Arguments:
            data: the data to be used to estimate the distribution
            lambtha: expected number of occurences in a given time frame
        """
        self.lambtha = lambtha
        self.data = data

    @property
    def data(self):
        """
        Gets the data of a poisson distribution
        Return:
            data
        """
        return self.__data

    @data.setter
    def data(self, value):
        """
        Sets the data of a poisson distribution
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
            self.__lambtha = sum(value) / len(value)

    @property
    def lambtha(self):
        """
        Gets the lambtha of a poisson distribution
        Return:
            lambtha
        """
        return self.__lambtha

    @lambtha.setter
    def lambtha(self, value):
        """
        Sets the lambtha of a poisson distribution
        Arguments:
            value: a positive number
        """
        if getattr(self, 'data', None) is not None:
            raise ValueError("cannot change lambtha unless data is None")
        if value <= 0:
            raise ValueError("lambtha must be a positive value")
        self.__lambtha = float(value)

    def pmf(self, k):   # pylint: disable=invalid-name
        """
        Calculates the value of the PMF for a given number of “successes”
        Arguments:
            k: the number of successes
        Return:
            If k is out of range, return 0.
            Otherwise, return the PMF value for k.
        """
        k = int(k)
        if k < 0:
            return 0
        fac = 1
        for num in range(k + 1):
            fac *= num or 1
        return self._e ** -self.lambtha * self.lambtha ** num / fac

    def cdf(self, k):   # pylint: disable=invalid-name
        """
        Calculates the value of the CDF for a given number of “successes”
        Arguments:
            k: the number of successes
        Return:
            If k is out of range, return 0.
            Otherwise, return the CDF value for k.
        """
        k = int(k)
        if k < 0:
            return 0
        cdf = 0
        fac = 1
        for num in range(k + 1):
            fac *= num or 1
            cdf += self._e ** -self.lambtha * self.lambtha ** num / fac
        return cdf
