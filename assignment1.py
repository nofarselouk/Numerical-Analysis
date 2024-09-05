"""
In this assignment you should interpolate the given function.
"""
import math

import numpy as np
import time
import random


class Assignment1:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        starting to interpolate arbitrary functions.
        """

        pass

    def coefficient(self, xv, yv, n):
        """
        xv: list containing n - x data points
        y: list containing n - y data points
        n: the amount of points
        returns: list with all the coefficients for the newton polynomial
        """

        x = np.copy(xv)
        coeff = np.copy(yv)
        for i in range(1, n):
            coeff[i:] = (coeff[i:] - coeff[i - 1]) / (x[i:] - x[i - 1])
        return coeff

    def newton_polynomial(self, xv, yv, x, n):
        """
        xv: list containing n - x data points
        yv: list containing n - y data points
        x: evaluation point(s)
        n: the amount of points
        returns: the estimated polynomial
        """
        coeff = self.coefficient(xv, yv, n)
        degree = n - 1  # Degree of polynomial
        p = coeff[degree]
        for i in range(1, degree + 1):
            p = coeff[degree - i] + (x - xv[degree - i]) * p
        return p

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Interpolate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the interpolation error.
        Your secondary objective is minimizing the running time. 
        The assignment will be tested on variety of different functions with 
        large n values. 
        
        Interpolation error will be measured as the average absolute error at 
        2*n random points between a and b. See test_with_poly() below. 

        Note: It is forbidden to call f more than n times. 

        Note: This assignment can be solved trivially with running time O(n^2)
        or it can be solved with running time of O(n) with some preprocessing.
        **Accurate O(n) solutions will receive higher grades.** 
        
        Note: sometimes you can get very accurate solutions with only few points, 
        significantly less than n. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        n : int
            maximal number of points to use.

        Returns
        -------
        The interpolating function.
        """
        x_values = []
        for i in range(n):  # The calculation of the Chebyshev points
            x_values.append((a + b) / 2 + (b - a) / 2 * np.cos((2 * i + 1) * np.pi / (2 * n)))
        y_values = [f(i) for i in x_values]
        P = (lambda x: self.newton_polynomial(x_values, y_values, x, n))
        return P

    ##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, -10, 10, 100)

            xs = np.random.random(200)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print(T)
        print(mean_err)

    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)


if __name__ == "__main__":
    unittest.main()
