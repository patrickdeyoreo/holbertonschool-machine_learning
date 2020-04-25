# 0x02 - Calculus

## Learning Objectives

- Summation and Product notation
- What is a series?
- Common series
- What is a derivative?
- What is the product rule?
- What is the chain rule?
- Common derivative rules
- What is a partial derivative?
- What is an indefinite integral?
- What is a definite integral?
- What is a double integral?

---

### [0. Sigma is for sum](./0-sigma_is_for_sum)

- Evaluate the provided summation (multiple choice)


### [1. It's actually pronounced sEEgma](./1-seegma)

- Evaluate the provided summation (multiple choice)


### [2. Pi is for product](./2-pi_is_for_product)

- Evaluate the provided product (multiple choice)


### [3. It's actually pronounced pEE](./3-pee)

- Evaluate the provided product (multiple choice)


### [4. Hello, derivatives!](./4-hello_derivatives)

- Evaluate the provided derivative (multiple choice)


### [5. A log on the fire](./5-log_on_fire)

- Evaluate the provided derivative (multiple choice)


### [6. It is difficult to free fools from the chains they revere](./6-voltaire)

- Evaluate the provided derivative (multiple choice)


### [7. Partial truths are often more insidious than total falsehoods](./7-partial_truths)

- Evaluate the provided derivative (multiple choice)


### [8. Put it all together and what do you get?](./8-all-together)

- Evaluate the provided derivative (multiple choice)


### [9. Our life is the sum total of all the decisions we make every day, and those decisions are determined by our priorities](./9-sum_total.py)

- Write a function `summation_i_squared(n)` that calculates the sum from `1` to `n` of `i^2`
- `n` is the stopping condition
- Return the integer value of the sum
- If `n` is not a valid number, return `None`
- You are not allowed to use any loops


### [10. Derive happiness in oneself from a good day's work](./10-matisse.py)

- Write a function `poly_derivative(poly)` that calculates the derivative of a polynomial.
- `poly` is a list of coefficients representing a polynomial:
  * The index of the list represents the power of `x` to which the coefficient belongs.
  * Example: If `f(x) = x^3 + 3x + 5` then poly is equal to `[5, 3, 0, 1]`.
- If `poly` is not valid, return `None`.
- If the derivative is `0`, return `[0]`.
- Otherwise return a new list of coefficients representing the derivative of the polynomial.


### [11. The Western Exchange](./11-the_western_exchange.py)
- Write a function `np_transpose(matrix)` that transposes matrix.


### [12. Bracing The Elements](./12-bracin_the_elements.py)
- Write a function `np_elementwise(mat1, mat2)` that performs element-wise addition, subtraction, multiplication, and division.


### [13. Cat's Got Your Tongue](./13-cats_got_your_tongue.py)
- Write a function `np_cat(mat1, mat2, axis=0)` that concatenates two matrices along a specific axis.


### [14. Saddle Up](./14-saddle_up.py)
- Write a function `np_matmul(mat1, mat2)` that performs matrix multiplication.


### [15. Slice Like A Ninja](./100-slice_like_a_ninja.py)
- Write a function `np_slice(matrix, axes={})` that slices a matrix along a specific axes.


### [16. The Whole Barn](./101-the_whole_barn.py)

- Write a function `add_matrices(mat1, mat2)` that adds two matrices.


### [17. Integrate](./17-integrate.py)

- Write a function `poly_integral(poly, C=0)` that calculates the integral of a polynomial.
- `poly` is a list of coefficients representing a polynomial:
  * The index of the list represents the power of `x` to which the coefficient belongs.
  * Example: If `f(x) = x^3 + 3x + 5` then poly is equal to `[5, 3, 0, 1]`.
- `C` is an integer representing the integration constant.
- If a coefficient is a whole number, it should be represented as an integer.
- If `poly` or `C` is not valid, return `None`.
- Otherwise return a new list of coefficients representing the integral of the polynomial.
- The returned list should be as small as possible.
---

## Author
* [**Patrick DeYoreo**](github.com/patrickdeyoreo)
