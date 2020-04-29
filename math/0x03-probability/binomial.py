#!/usr/bin/env python3
"""Provides a class ``Binomial'' to represent a binomial distribution"""


class Binomial:
    """Represents a binomial distribution"""

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

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initializes a binomial distribution
        Arguments:
            data: the data to be used to estimate the distribution
            n: the number of Bernoulli trials
            p: is the probability of a “success”
        """
        self.n = n  # pylint: disable=invalid-name
        self.p = p  # pylint: disable=invalid-name
        self.data = data

    @property
    def data(self):
        """
        Gets the data of a binomial distribution
        Return:
            data
        """
        return self.__data

    @data.setter
    def data(self, value):
        """
        Sets the data of a binomial distribution
        Arguments:
            value: a list containing at least two data points (or None)
        """
        if value is None:
            self.__data = None
        elif isinstance(value, list) is False:
            raise TypeError("data must be a list")
        elif len(value) < 2:
            raise ValueError("data must contain multiple values")
        else:
            self.__data = value[:]
            mean = sum(self.data) / len(self.data)
            variance = sum((x - mean) ** 2 for x in self.data) / len(self.data)
            self.__p = 1 - variance / mean
            self.__n = round(mean / self.p)
            self.__p = sum(x / self.n for x in self.data) / len(self.data)
            p = 1 - variance / mean

    @property
    def n(self):     # pylint: disable=invalid-name
        """
        Gets the number of Bernoulli trials of a binomial distribution
        Return:
            n
        """
        return self.__n

    @n.setter
    def n(self, value):     # pylint: disable=invalid-name
        """
        Sets the number of Bernoulli trials of a binomial distribution
        Arguments:
            value: a positive integer
        """
        if getattr(self, 'data', None) is not None:
            raise ValueError("cannot change n unless data is None")
        if value <= 0:
            raise ValueError("n must be a positive value")
        self.__n = round(value)

    @property
    def p(self):    # pylint: disable=invalid-name
        """
        Gets the probability of a “success" of a binomial distribution
        Return:
            stddev
        """
        return self.__p

    @p.setter
    def p(self, value):     # pylint: disable=invalid-name
        """
        Sets the probability of a “success" of a binomial distribution
        Arguments:
            value: a probability
        """
        if getattr(self, 'data', None) is not None:
            raise ValueError("cannot change p unless data is None")
        if 0 <= value <= 1 is False:
            raise ValueError("p must be greater than 0 and less than 1")
        self.__p = float(value)

    # def z_score(self, x):   # pylint: disable=invalid-name
    #     """
    #     Calculates the z-score of a given x-value
    #     Arguments:
    #         x: the x-value
    #     Return:
    #         the z-score of x
    #     """
    #     return (x - self.mean) / self.stddev

    # def x_value(self, z):   # pylint: disable=invalid-name
    #     """
    #     Calculates the x-value of a given z-score
    #     Arguments:
    #         z: the z-score
    #     Return:
    #         the x-value of z
    #     """
    #     return z * self.stddev + self.mean

    # def pdf(self, x):   # pylint: disable=invalid-name
    #     """
    #     Calculates the value of the PDF for a given x-value
    #     Arguments:
    #         x: the x-value
    #     Return:
    #         the PDF value for x
    #     """
    #     numer = self._e ** (-0.5 * ((x - self.mean) / self.stddev) ** 2)
    #     denom = self.stddev * (2 * self._pi) ** 0.5
    #     return numer / denom

    # def cdf(self, x):   # pylint: disable=invalid-name
    #     """
    #     Calculates the value of the CDF for a given x-value
    #     Arguments:
    #         x: the x-value
    #     Return:
    #         the CDF value for x
    #     """
    #     return (1 + self._erf((x - self.mean) / (self.stddev * 2 ** 0.5))) / 2
