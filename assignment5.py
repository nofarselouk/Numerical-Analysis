"""
In this assignment you should fit a model function of your choice to data 
that you sample from a contour of given shape. Then you should calculate
the area of that shape. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you know that your iterations may take more 
than 1-2 seconds break out of any optimization loops you have ahead of time.

Note: You are allowed to use any numeric optimization libraries and tools you want
for solving this assignment. 
Note: !!!Despite previous note, using reflection to check for the parameters 
of the sampled function is considered cheating!!! You are only allowed to 
get (x,y) points from the given shape by calling sample(). 
"""

import numpy as np
import time
import random
import math

from commons import shape3, shape5
from functionUtils import AbstractShape
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt


class MyShape(AbstractShape):
    # change this class with anything you need to implement the shape
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def area(self):
        area = 0.5 * np.abs(np.dot(self.x, np.roll(self.y, 1)) - np.dot(self.y, np.roll(self.x, 1)))
        return np.float32(area)


class Assignment5:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def Shoe_lace(self, x, y):
        """
        The implementation of the Shoelace method that calculate the area of closed shape

        Parameters:
            x : all the X points of the shape
            y : all the y points of the shape

        Returns:
            the area of the shape
        """
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        return np.float32(area)

    def area(self, contour: callable, maxerr=0.001) -> np.float32:
        """
        Compute the area of the shape with the given contour.

        Parameters
        ----------
        contour : callable
            Same as AbstractShape.contour
        maxerr : TYPE, optional
            The target error of the area computation. The default is 0.001.

        Returns
        -------
        The area of the shape.

        """
        n = 1000
        points = contour(n)
        result = 0
        res = 0
        x = []
        y = []
        for i in range(len(points)):
            x.append(points[i][0])
            y.append(points[i][1])
        res += self.Shoe_lace(x, y)
        while n <= 10000:
            n = int(n * 1.5)
            points = contour(n)
            x = []
            y = []
            for i in range(len(points)):
                x.append(points[i][0])
                y.append(points[i][1])
            result += self.Shoe_lace(x, y)
            if (abs(result - res) / result) <= maxerr:
                return np.float32(result)
            else:
                res = result
                result = 0
        return np.float(res)

    def clockwise_sort(self, points):
        """
        Implementation of the clockwise sort that sort the points by their center point

        Parameters:
            points : all the shape points

        Returns:
            numpy array with all the points sorted by the method
        """
        center = (sum(x for x, y in points) / len(points),
                  sum(y for x, y in points) / len(points))
        points.sort(key=lambda p: math.atan2(p[1] - center[1], p[0] - center[0]))
        return np.array(points)

    def polygon_area(self, points):
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        return 0.5 * abs(
            sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(len(points) - 1)) + x[-1] * y[0] - x[0] * y[-1])

    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. 
        
        Parameters
        ----------
        sample : callable. 
            An iterable which returns a data point that is near the shape contour.
        maxtime : float
            This function returns after at most maxtime seconds. 

        Returns
        -------
        An object extending AbstractShape. 
        """

        # replace these lines with your solution
        n = 100000
        samples = [sample() for i in range(n)]
        points = self.clockwise_sort(samples)
        arrays = splprep(points.T, u=None, s=0.0, per=1)
        tck = arrays[0]
        u = arrays[1]
        u_new = np.linspace(min(u), max(u), 1000)
        x_new, y_new = splev(u_new, tck, der=0)
        return MyShape(x_new, y_new)


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment5(unittest.TestCase):

    def test_area(self):
        ass5 = Assignment5()
        T = time.time()
        ar = ass5.area(shape3().contour, 0.001)
        print(ar)
        print(shape5().area())
        print(shape5().area() - ar)

    def test_return(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    def test_delay(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)

        def sample():
            time.sleep(7)
            return circ()

        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=sample, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertGreaterEqual(T, 5)

    def test_circle_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_bezier_fit(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)


if __name__ == "__main__":
    unittest.main()
