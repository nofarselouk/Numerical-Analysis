"""
In this assignment you should find the intersection points for two functions.
"""

import numpy as np
import time
import random
import math
from scipy.misc import derivative
from collections.abc import Iterable

from commons import f10, f6, f2_nr, f3_nr


class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def area_finder(self, func, a, b, error):
        """
        The function detect where in the range there is a root and returns it

        Parameters:
            func : the function we need to find roots for
            a : the start of the rang
            b : the end of the range
            error : the max error the root can contain

        Returns :
            The range that contain root
        """
        # detect where in the range there is a root and returns the range
        x1 = a
        fx1 = func(a)
        x2 = a + error
        fx2 = func(x2)
        while fx1 * fx2 > 0.0:
            if x1 >= b:
                return None, None
            x1 = x2
            fx1 = fx2
            x2 = x1 + error
            fx2 = func(x2)
        return x1, x2

    def Regula_Falsi(self, func, a, b, error):
        """
        The method to find roots Regula Falsi

        Parameters:
           func : the function we need to find roots for
            a : the start of the rang
            b : the end of the range
            error : the max error the root can contain

        Returns :
            The root of the function in the range [a,b]
        """
        func_a, func_b = func(a), func(b)
        if func_a == 0:
            return a
        if func_b == 0:
            return b
        if func_a * func_b > 0:
            return None
        c = a - func_a * ((a - b) / (func_a - func_b))
        func_c = func(c)
        while abs(func_c) > error:
            if (abs(func_c) > abs(func_a)) and (abs(func_c) > abs(func_b)):  # if the func is out of range
                return None
            if func_c == 0:
                return c
            if func_c * func_b < 0:
                a = c
                func_a = func_c
            else:
                b = c
                func_b = func_c
            c = a - func_a * ((a - b) / (func_a - func_b))
            func_c = func(c)
        return c

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.
        
        This function may not work correctly if there is infinite number of
        intersection points. 


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """
        res = []
        if type(f1) == int:
            func = (lambda x: f1 - f2(x))
        elif type(f2) == int:
            func = (lambda x: f1(x) - f2)
        elif type(f1) == int and type(f2) == int:
            return res
        else:
            func = (lambda x: f1(x) - f2(x))
        while a < b:
            x1, x2 = self.area_finder(func, a, b, maxerr)
            if x1 is not None:
                a = x2
                root = self.Regula_Falsi(func, x1, x2, maxerr)
                if root is not None:
                    res.append(root)
                    pass
            else:
                break
        return res


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment2(unittest.TestCase):

    def test_sqr(self):

        ass2 = Assignment2()

        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)
        print(X)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(10)

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)
        print(X)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_func(self):

        ass2 = Assignment2()
        f1 = np.poly1d([1.382, -0.9652, -1.511, 1.271, -3.769, 1.139, -0.1816, -1.581, 0.08984, -0.01779, -0.8179])
        f2 = 0

        X = ass2.intersections(f1, f2, -2, 2, maxerr=0.005)
        print(X)

        for x in X:
            self.assertGreaterEqual(0.005, abs(f1(x) - f2))

    def test_func1(self):

        ass2 = Assignment2()
        f1 = (lambda x: np.sin(4 * x) * 2 + x + 3)
        f2 = (lambda x: np.cos(x / 6) * x)

        X = ass2.intersections(f1, f2, -10, 10, maxerr=0.005)
        print(X)

        for x in X:
            self.assertGreaterEqual(0.005, abs(f1(x) - f2(x)))


if __name__ == "__main__":
    unittest.main()
