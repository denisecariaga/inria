import numpy as np
from time import time
from math import sqrt
from scipy.sparse import linalg, csr_matrix, csc_matrix


class Data:
    def __init__(self):
        self.W = np.ones(3)
        self.r = np.array([0, 0, 0])
        self.M = np.ones(3)
        self.H = np.ones(3)
        self.q = np.array([0, 0, 0])
        self.f = np.array([0, 0, 0])
        self.s = np.array([0, 0, 0])
        self.n = np.size(self.f)
        self.m = np.size(self.r)
        self.nc = self.m / 3
        self.g = 10**(-6)


class APGDMethod:
    def __init__(self, r, W, q, m, n_c, mu, rho, g):
        self.r = r
        self.W = W
        self.q = q
        self.m = m
        self.n_c = n_c          # m/3
        self.mu = mu
        self.dim1 = 3
        self.rho = rho
        self.g = g

    def project(self, vector):
        vector_per_contact = np.split(vector, self.n_c)
        projected = np.array([])

        for i in range(int(self.n_c)):
            mui = self.mu[i]
            x1 = vector_per_contact[i][0]
            norm2 = np.linalg.norm(vector_per_contact[i][1:])

            if norm2 <= (-1 / mui) * x1:
                projected = np.concatenate((projected, np.zeros([self.dim1, ])))
            elif norm2 <= mui * x1:
                projected = np.concatenate((projected, vector_per_contact[i]))
            else:
                x2 = vector_per_contact[i][1:]
                projected = np.concatenate((projected,
                                            (1 / (1 + mui ** 2)) * (x1 + mui * norm2) * np.concatenate(
                                                (np.array([1]), mui * x2 * (1 / norm2)))))

    def accelerate(self, k):
        return self.r[k-1] + ((k - 2) / (k + 1)) * (self.r[k-1] - self.r[k-2])

    def update_r(self, k):
        self.r[k].append(self.project(
            self.accelerate(k) - self.rho * (csr_matrix.dot(self.W,  self.accelerate(k)) + self.q)))

    def stop_criteria(self, k, epsilon):
        res = (1 / (self.m * self.g)) * (self.r[k] - self.project(
            self.r[k] - self.g * (csr_matrix.dot(self.W,  self.accelerate(k)) + self.q)))
        norm_res = np.linalg.norm(res.toarray(), ord=2)
        if norm_res < epsilon:
            return True
        else:
            return False


class Rho:
    def __init__(self, W, M, H, tau, d):
        self.W = W
        self.M = M
        self.H = H
        self.tau = tau
        self.d = d

    def normal_rho(self):
        return 1

    def smaller_rho(self):
        return 2 / 3

    def w_rho(self):
        return 1 / np.linalg.norm(self.W)

    def eigen_w_rho(self):
        eig, eigv = linalg.eigs(self.W)
        eig_max = np.absolute(np.amax(eig))
        return 1 / eig_max

    def ghadimi_rho(self):
        eig, eig_v = linalg.eigs(self.W)
        eig_max = np.absolute(np.amax(eig))
        eig_min = np.absolute(np.min(eig[np.nonzero(eig)]))
        return 1 / sqrt(eig_max * eig_min)

    def di_cairamo_rho(self):
        eig, eig_v = linalg.eigs(self.M)
        eig_max = np.absolute(np.amax(eig))
        eig_min = np.absolute(np.min(eig[np.nonzero(eig)]))
        return sqrt(eig_max * eig_min)

    def acary_rho(self):
        return np.linalg.norm(self.M.toarray(), ord=1) / np.linalg.norm(self.H.toarray(), ord=1)

