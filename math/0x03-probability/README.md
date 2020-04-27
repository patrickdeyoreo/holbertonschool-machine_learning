# 0x03 - Probability

## Learning Objectives

- What is probability?
- Basic probability notation
- What is independence? What is disjoint?
- What is a union? intersection?
- What are the general addition and multiplication rules?
- What is a probability distribution?
- What is a probability distribution function? probability mass function?
- What is a cumulative distribution function?
- What is a percentile?
- What is mean, standard deviation, and variance?
- Common probability distributions

---

### [0. Initialize Poisson](./poisson.py)

- Create a class `Poisson` that represents a poisson distribution:
  - Class contructor `__init__(self, data=None, lambtha=1.)`:
    - `data` is a list of the data to be used to estimate the distribution
    - `lambtha` is the expected number of occurences in a given time frame
    - Sets the instance attribute `lambtha`
      - Saves `lambtha` as a `float`
    - If `data` is not given, (i.e. `None`):
      - Use the given `lambtha`
      - If `lambtha` is not a positive value, raise a `ValueError` with the message `lambtha must be a positive value`
    - If `data` is given:
      - Calculate the `lambtha` of `data`
      - If `data` is not a `list`, raise a `TypeError` with the message `data must be a list`
      - If `data` does not contain at least two data points, raise a `ValueError` with the message `data must contain multiple values`

---

## Author

[**Patrick DeYoreo**](github.com/patrickdeyoreo)
