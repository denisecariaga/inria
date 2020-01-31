'''
% Solves the following problem via ADMM:
%
%   minimize     (1/2)*r'*W*r + r'*(q + s) + indicator_K(p)
%   subject to   Id*r - Id*p = 0
'''
######################
## IMPORT LIBRARIES ##
######################

# Math libraries
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import linalg
import matplotlib.pyplot as plt

# Timing
import time

# Import data
from ADMM.data import *
# from Data.read_fclib import *

# Initial penalty parameter
import Solver.Rho.Optimal

# Max iterations and kind of tolerance
from Solver.Tolerance.iter_totaltolerance import *

# b = Es matrix
from Data.Es_matrix import *

# Projection onto second order cone
from Solver.ADMM_iteration.Numerics.projection import *


# problem data is a class with all the parameters needed to solve the FCP


class ADMM:
    def __init__(self, problem_data, rho_method):
        #EXTRACTING PROBLEM INFORMATION
        self.pd = problem_data()
        self.rho_method = rho_method

        #TIMES
        self.start = time.clock()
        self.end = time.clock()

        #RESIDUALS
        self.r_prim = r_prim
        self.r_dual = r_dual

        #TITLE
        self.title = title

        #KNOWN PARAMETERS
        self.mu = self.pd.mu
        self.M = self.pd.M
        self.f = self.pd.f
        self.w = self.pd.w

        #PROBLEM DIMENSIONS
        self.dim1 = 3
        self.m = self.pd.m
        self.n = self.pd.n

        #UNKNOWN VARIABLES

    def plot(self):
        R = [np.linalg.norm(k) for k in self.r_prim]
        S = [np.linalg.norm(k) for k in self.r_dual]
        plt.semilogy(R, label='||r||')
        plt.hold(True)
        plt.semilogy(S, label='||s||')
        plt.hold(True)
        plt.ylabel('Residuals')
        plt.xlabel('Iteration')
        plt.text(len(self.r_prim) / 2, np.log(np.amax(S) + np.amax(R)) / 10, 'N_iter = ' + str(len(self.r_prim) - 1))
        plt.text(len(self.r_prim) / 2, np.log(np.amax(S) + np.amax(R)) / 100,
                 'Total time = ' + str((self.end - self.start) * 10 ** 3) + ' ms')
        plt.text(len(self.r_prim) / 2, np.log(np.amax(S) + np.amax(R)) / 1000,
                 'Time_per_iter = ' + str(((self.end - self.start) / (len(self.r_prim) - 1)) * 10 ** 3) + ' ms')
        plt.title(self.title)
        plt.legend()
        plt.show()

    def projection(self, vector):
        vector_per_contact = np.split(vector, self.m / self.dim1)
        projected = np.array([])

        for i in range(int(self.m / self.dim1)):
            mui = self.mu[i]
            x1 = vector_per_contact[i][0]
            normx2 = np.linalg.norm(vector_per_contact[i][1:])

            if normx2 <= (-mui) * x1:
                projected = np.concatenate((projected, np.zeros([self.dim1, ])))
            elif normx2 <= (1 / mui) * x1:
                projected = np.concatenate((projected, vector_per_contact[i]))
            else:
                x2 = vector_per_contact[i][1:]
                projected = np.concatenate((projected,
                                            (mui ** 2) / (1 + mui ** 2) * (x1 + (1 / mui) * normx2) * np.concatenate(
                                                (np.array([1]), (1 / mui) * x2 * (1 / normx2)))))

        return projected

    def
