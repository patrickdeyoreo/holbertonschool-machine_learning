#!/usr/bin/env python3

poly_integral = __import__('17-integrate').poly_integral

polynomials = [
    [5, 3, 0, 1],
    [1, 0, 0],
    [0, 0],
    ['a'],
    [],
    None,
]

constants = [
    0,
    [1],
    None,
]

for i, poly in enumerate(polynomials):
    functions = ['poly_integral({}, C={})'.format(poly, c) for c in constants]
    padding = max(len(f) for f in functions)
    for func, const in zip(functions, constants):
        print('{{:<{}}} = {{}}'.format(padding).format(
            func, poly_integral(poly, C=const)))
    if i + 1 < len(polynomials):
        print()
