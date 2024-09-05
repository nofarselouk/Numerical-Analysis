"""
In this assignment you should find the area enclosed between the two given functions.
The rightmost and the leftmost x values for the integration are the rightmost and 
the leftmost intersection points of the two functions. 

The functions for the numeric answers are specified in MOODLE. 


This assignment is more complicated than Assignment1 and Assignment2 because: 
    1. You should work with float32 precision only (in all calculations) and minimize the floating point errors. 
    2. You have the freedom to choose how to calculate the area between the two functions. 
    3. The functions may intersect multiple times. Here is an example: 
        https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx
    4. Some of the functions are hard to integrate accurately. 
       You should explain why in one of the theoretical questions in MOODLE. 

"""

import numpy as np
import time
import random

import assignment1
import assignment2
import random

from commons import f10, f6, f2


class Assignment3:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def simpson(self, f, a, b, n):
        """
        The method Simpson rule for integration in python

        Parameters:
            f : the function we need to calculate the integrate
            a : the start of the range
            b : the end of the range
            n : the amount of points we are allowed to use

        Returns :
            the integrate of the function in the range [a,b]
        """
        h = (b - a) / n
        F02 = 0
        F1 = 0
        for i in range(1, n):
            x = a + i * h
            if i % 2 == 0:
                F02 += f(x)
            else:
                F1 += f(x)
        res = np.float32((h / 3) * (4 * F1 + 2 * F02 + f(a) + f(b)))

        return res

    def integrate(self, f: callable, a: float, b: float, n: int) -> np.float32:
        """
        Integrate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the integration error. 
        Your secondary objective is minimizing the running time. The assignment
        will be tested on variety of different functions. 
        
        Integration error will be measured compared to the actual value of the 
        definite integral. 
        
        Note: It is forbidden to call f more than n times. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the integration range.
        b : float
            end of the integration range.
        n : int
            maximal number of points to use.

        Returns
        -------
        np.float32
            The definite integral of f between a and b
        """
        if (n % 2) == 0:
            res = self.simpson(f, a, b, n - 2)
        else:
            res = self.simpson(f, a, b, n - 1)
        return res

    def areabetween(self, f1: callable, f2: callable) -> np.float32:
        """
        Finds the area enclosed between two functions. This method finds 
        all intersection points between the two functions to work correctly. 
        
        Example: https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx

        Note, there is no such thing as negative area. 
        
        In order to find the enclosed area the given functions must intersect 
        in at least two points. If the functions do not intersect or intersect 
        in less than two points this function returns NaN.  
        This function may not work correctly if there is infinite number of 
        intersection points. 
        

        Parameters
        ----------
        f1,f2 : callable. These are the given functions

        Returns
        -------
        np.float32
            The area between function and the X axis

        """
        ass2 = assignment2.Assignment2()
        intersections = list(ass2.intersections(f1, f2, 1, 100))
        result = 0
        if len(intersections) == 0:
            return np.float32(0)
        if len(intersections) == 1:
            if abs(intersections[0] - 1) < abs(intersections[0] - 100):
                intersections.insert(0, 1)
            else:
                intersections.append(100)
        for i in range(1, len(intersections)):
            x = abs(intersections[i] - intersections[i - 1]) / 2
            if f1(x) >= f2(x):
                result += abs(self.integrate(lambda y: f1(y) - f2(y), intersections[i - 1], intersections[i], 500))
            if f1(x) < f2(x):
                result += abs(self.integrate(lambda y: f2(y) - f1(y), intersections[i - 1], intersections[i], 500))
        return np.float32(result)


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment3(unittest.TestCase):

    def test_integrate_float32(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])
        r = ass3.integrate(f1, -1, 1, 9)
        print(r)
        self.assertEquals(r.dtype, np.float32)

    def test_integrate_hard_case(self):
        ass3 = Assignment3()
        f1 = strong_oscilations()
        r = ass3.integrate(f1, 0.09, 10, 20)
        print(r)
        true_result = -7.78662 * 10 ** 33
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    def test_areabetween1(self):
        ass3 = Assignment3()
        f1 = f10
        f2 = f6
        r = ass3.areabetween(f1, f2)
        print(f'sin_cos: {r}')
        true_result = 14.0539
        self.assertGreaterEqual(true_result / 10, abs(r - true_result))

    def test_areabetween2(self):
        ass3 = Assignment3()
        f1 = f10
        f23 = f2
        r = ass3.areabetween(f1, f23)
        print(f'sin_cos: {r}')
        true_result = 0.731
        self.assertGreaterEqual(true_result / 10, abs(r - true_result))

    def test_areasin(self):
        ass3 = Assignment3()
        f2 = lambda x: np.sin(x)
        f1 = lambda x: 0
        r = ass3.areabetween(f1, f2)
        print(r)
        true_result = 60
        print(true_result)
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))


if __name__ == "__main__":
    unittest.main()
