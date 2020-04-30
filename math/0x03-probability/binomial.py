#!/usr/bin/env python3
"""Provides a class ``Binomial'' to represent a binomial distribution"""


class Binomial:
    """Represents a binomial distribution"""

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
            p
        """
        return self.__p

    @p.setter
    def p(self, value):     # pylint: disable=invalid-name
        """
        Sets the probability of a “success" of a binomial distribution
        Arguments:
            value: a probability (0 < value < 1)
        """
        if getattr(self, 'data', None) is not None:
            raise ValueError("cannot change p unless data is None")
        if not 0 < value < 1:
            raise ValueError("p must be greater than 0 and less than 1")
        self.__p = float(value)

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
        if not 0 <= k <= self.n:
            return 0
        fac = num = 1
        while num < min(k, self.n - k):
            num += 1
            fac *= num
        denom = fac
        while num < max(k, self.n - k):
            num += 1
            fac *= num
        denom *= fac
        while num < self.n:
            num += 1
            fac *= num
        numer = fac
        return numer / denom * self.p ** k * (1 - self.p) ** (self.n - k)

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
        if not 0 <= k <= self.n:
            return 0
        return sum(self.pmf(x) for x in range(k + 1))
