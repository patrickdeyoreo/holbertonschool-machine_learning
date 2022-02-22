#!/usr/bin/env python3
"""Provides a class ``Normal'' to represent a normal distribution"""


class Normal:
    """Represents a normal distribution"""

    _pi = 3.1415926536
    _e = 2.7182818285

    @classmethod
    def _erf(cls, x):   # pylint: disable=invalid-name
        """
        Approximates the Gauss error function
        Arguments:
            x: the x-value
        Return:
            the probability that a random variable falls between −x and x
        """
        return (2/cls._pi**0.5) * (x - x**3/3 + x**5/10 - x**7/42 + x**9/216)

    def __init__(self, data=None, mean=0.0, stddev=1.0):
        """
        Initializes a normal distribution
        Arguments:
            data: the data to be used to estimate the distribution
            mean: the mean value of the data
            stddev: the standard deviation of the data
        """
        self.mean = mean
        self.stddev = stddev
        self.data = data

    @property
    def data(self):
        """
        Gets the data of a normal distribution
        Return:
            data
        """
        return self.__data

    @data.setter
    def data(self, value):
        """
        Sets the data of a normal distribution
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
            self.__mean = sum(self.data) / len(self.data)
            sum_squares = sum((x - self.mean) ** 2 for x in self.data)
            self.__stddev = (sum_squares / len(self.data)) ** 0.5

    @property
    def mean(self):
        """
        Gets the mean of a normal distribution
        Return:
            mean
        """
        return self.__mean

    @mean.setter
    def mean(self, value):
        """
        Sets the mean of a normal distribution
        Arguments:
            value: a number
        """
        if getattr(self, 'data', None) is not None:
            raise ValueError("cannot change mean unless data is None")
        self.__mean = float(value)

    @property
    def stddev(self):
        """
        Gets the stddev of a normal distribution
        Return:
            stddev
        """
        return self.__stddev

    @stddev.setter
    def stddev(self, value):
        """
        Sets the stddev of a normal distribution
        Arguments:
            value: a positive number
        """
        if getattr(self, 'data', None) is not None:
            raise ValueError("cannot change stddev unless data is None")
        if value <= 0:
            raise ValueError("stddev must be a positive value")
        self.__stddev = float(value)

    def z_score(self, x):   # pylint: disable=invalid-name
        """
        Calculates the z-score of a given x-value
        Arguments:
            x: the x-value
        Return:
            the z-score of x
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):   # pylint: disable=invalid-name
        """
        Calculates the x-value of a given z-score
        Arguments:
            z: the z-score
        Return:
            the x-value of z
        """
        return z * self.stddev + self.mean

    def pdf(self, x):   # pylint: disable=invalid-name
        """
        Calculates the value of the PDF for a given x-value
        Arguments:
            x: the x-value
        Return:
            the PDF value for x
        """
        numer = self._e ** (-0.5 * ((x - self.mean) / self.stddev) ** 2)
        denom = self.stddev * (2 * self._pi) ** 0.5
        return numer / denom

    def cdf(self, x):   # pylint: disable=invalid-name
        """
        Calculates the value of the CDF for a given x-value
        Arguments:
            x: the x-value
        Return:
            the CDF value for x
        """
        return (1 + self._erf((x - self.mean) / (self.stddev * 2 ** 0.5))) / 2
