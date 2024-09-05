"""
In this assignment you should fit a model function of your choice to data 
that you sample from a given function. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you take an iterative approach and know that 
your iterations may take more than 1-2 seconds break out of any optimization 
loops you have ahead of time.

Note: You are NOT allowed to use any numeric optimization libraries and tools 
for solving this assignment. 

"""

import numpy as np
import time
import random


class Assignment4:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def create_matrix(self, sample_x, d):
        """
        A function that create the matrix that represent the equations system coefficients

        Parameters:
            sample_x : all the X points we sampled
            d : the degree of the polynomial

        Returns:
            the matrix that represent the equations system coefficients
        """
        powers = np.array([sample_x ** i for i in range(d + 1)]).T
        matrix = np.zeros((d + 1, d + 1))
        for i in range(d + 1):
            for j in range(d + 1):
                matrix[i, j] = np.sum(powers[:, i] * powers[:, j])
        return matrix

    def answer_matrix(self, sample_x, sample_y, d):
        """
        A function that builds the answer vector of the equations system

        Parameters:
            sample_x : all the X points we sampled
            sample_y : the Y points we sampled from the function
            d : the degree of the polynomial

        Returns:
            the answer vector of the equations system

        """
        answer = np.array([])
        for i in range(d + 1):
            answer = np.append(answer, np.dot(sample_x ** i, sample_y))
        return answer

    def luDecomposition(self, mat):
        """
        The method Lu Decomposition that decompose a matrix to lower and upper matrix

        Parameters:
            mat : the matrix we want to decompose

        Return:
            the lower and upper matrix
        """
        n = len(mat)
        lower = np.zeros((n, n))
        upper = np.zeros((n, n))
        for i in range(n):
            lower[i][i] = 1

        for k in range(n):
            upper[k][k:] = mat[k][k:] - lower[k, :k] @ upper[:k, k:]
            lower[(k + 1):, k] = (mat[(k + 1):, k] - lower[(k + 1):, :k] @ upper[:k, k]) / upper[k, k]

        return lower, upper

    def gauss(self, l_mat, a_mat):
        """
        Gauss elimination method to solve a system of equations

        Parameters:
            l_mat : a matrix that contain the coefficients of equations system
            a_mat : the vector that contain the solutions of the equations system

        Returns:
            a vector with the solutions
        """
        n = len(l_mat)

        # Forward elimination
        for k in range(n - 1):
            factor = l_mat[k + 1:n, k] / l_mat[k, k]
            l_mat[k + 1:n, k + 1:n] -= np.outer(factor, l_mat[k, k + 1:n])
            a_mat[k + 1:n] -= factor * a_mat[k]

        # Backward substitution
        x = np.zeros(n)
        x[n - 1] = a_mat[n - 1] / l_mat[n - 1, n - 1]
        for i in range(n - 2, -1, -1):
            x[i] = (a_mat[i] - np.dot(l_mat[i, i + 1:], x[i + 1:])) / l_mat[i, i]
        return x

    def fit(self, f: callable, a: float, b: float, d: int, maxtime: float) -> callable:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape.

        Parameters
        ----------
        f : callable.
            A function which returns an approximate (noisy) Y value given X.
        a: float
            Start of the fitting range
        b: float
            End of the fitting range
        d: int
            The expected degree of a polynomial matching f
        maxtime : float
            This function returns after at most maxtime seconds.

        Returns
        -------
        a function:float->float that fits f between a and b
        """
        T = time.time()
        f((a + b))
        T1 = time.time()
        time_per_sample = T1 - T
        if time_per_sample > maxtime:
            return
        elif time_per_sample == 0:
            time_per_sample = 0.0001
        elif time_per_sample <= 0.0005:
            time_per_sample = 0.0001
        else:
            time_per_sample = time_per_sample * 1.1
        samp_range = int((maxtime - 1.5) / time_per_sample)
        sample_x = np.linspace(a, b, samp_range)
        sample_y = [f(x) for x in sample_x]
        main_matrix = self.create_matrix(sample_x, d)
        ans_matrix = self.answer_matrix(sample_x, sample_y, d)
        l_matrix, u_matrix = self.luDecomposition(main_matrix)
        z = self.gauss(l_matrix, ans_matrix)
        coeff = self.gauss(u_matrix, z)
        poly_ans = np.poly1d(np.flip(coeff))
        return poly_ans


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment4(unittest.TestCase):

    def test_return(self):
        f = NOISY(0.01)(poly(1, 1, 1))
        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertLessEqual(T, 5)

    def test_delay(self):
        f = DELAYED(7)(NOISY(0.01)(poly(1, 1, 1)))

        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertGreaterEqual(T, 5)

    def test_err(self):
        f = poly(1, 1, 1)
        nf = NOISY(1)(f)
        ass4 = Assignment4()
        T = time.time()
        ff = ass4.fit(f=nf, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        mse = 0
        for x in np.linspace(0, 1, 1000):
            self.assertNotEquals(f(x), nf(x))
            mse += (f(x) - ff(x)) ** 2
        mse = mse / 1000
        print(mse)


if __name__ == "__main__":
    unittest.main()
