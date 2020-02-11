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

# Initial penalty parameter
import ADMM.Solver.Rho

# Max iterations and kind of tolerance
from ADMM.Solver.Tolerance.iter_tolerance import *

# b = Es matrix




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
        self.r_prim = self.pd.r_prim
        self.r_dual = self.pd.r_dual

        #TITLE
        self.title = self.pd.title

        #KNOWN PARAMETERS
        self.mu = self.pd.mu
        self.M = self.pd.M
        self.f = self.pd.f
        self.w = self.pd.w
        self.W = self.pd.W
        self.q = self.pd.q
        self.Id = self.pd.Id

        #PROBLEM DIMENSIONS
        self.dim1 = 3
        self.m = self.pd.m
        self.n = self.pd.n
        self.n_c = self.m / self.dim1

        #UNKNOWN VARIABLES
        self.u = self.pd.u
        self.r = self.pd.r
        self.p = self.pd.p
        self.s = self.pd.s
        self.x = self.pd.x

        #OTHERS
        A = self.W + self.rho_method * csc_matrix(self.Id)
        self.LU = linalg.splu(A)

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
        vector_per_contact = np.split(vector, self.n_c)
        projected = np.array([])

        for i in range(int(self.n_c)):
            mui = self.mu[i]
            x1 = vector_per_contact[i][0]
            normx2 = np.linalg.norm(vector_per_contact[i][1:])

            if normx2 <= (-1/mui) * x1:
                projected = np.concatenate((projected, np.zeros([self.dim1, ])))
            elif normx2 <= mui * x1:
                projected = np.concatenate((projected, vector_per_contact[i]))
            else:
                x2 = vector_per_contact[i][1:]
                projected = np.concatenate((projected,
                                            (1/ (1 + mui ** 2)) * (x1 + mui * normx2) * np.concatenate(
                                                (np.array([1]), mui * x2 * (1 / normx2)))))

        return projected

    def Gs_matrix(self):
        E_ = np.array([])

        u_per_contact = np.split(self.u, self.n_c)

        for i in range(self.n_c):
            E_ = np.concatenate(E_, np.array([1, 0, 0]) * self.mu[i] * np.linalg.norm(u_per_contact[i][1:]))
        E = E_[:, np.newaxis]

        return np.squeeze(E)

    def r_update(self, k):
        RHS = -self.q + self.s + self.rho_method * csc_matrix(self.p[k] - self.x[k])
        self.r.append(self.LU.solve(RHS))
        return self.r

    def p_update(self, k):
        vector = self.r[k] + self.x[k]
        self.p.append(self.projection(vector))
        return self.p

    def x_update(self, k):
        self.x.append(self.x[k] + self.r[k+1] - self.p[k+1])
        return self.x

    def residuals_update(self, k):
        self.r_prim.append(self.r[k+1] - self.p[k+1])
        self.r_dual.append(self.rho_method * csc_matrix.dot(self.Id, (self.p[k] - self.p[k+1])))

    def stop_criteria(self):




