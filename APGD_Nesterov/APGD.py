import numpy as np
from time import time
from math import sqrt
from scipy.sparse import linalg, csr_matrix, csc_matrix


class APGD:
    def __init__(self, r, W, q, n_c, mu, rho):
        self.r = r
        self.W = W
        self.q = q
        self.n_c = n_c          # m/3
        self.mu = mu
        self.dim1 = 3
        self.rho = rho

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
        aqui hacer bien la multiplicacion matricial de W*v
        self.r[k].append(self.project(self.accelerate(k) - self.rho * (self.W * self.accelerate(k) + self.q)))


class Datos:
    def __init__(self):
        self.W = np.ones(3)


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
        return 1 / max(linalg.eigs(self.W))

    def ghadimi_rho(self):
        return 1 / sqrt(max(linalg.eigs(self.W)) * min(linalg.eigs(self.W)))

    def di_cairamo_rho(self):
        return sqrt(min(linalg.eigs(self.M)) * max(linalg.eigs(self.M)))

    def acary_rho(self):
        return np.linalg.norm(self.M, ord = 1) / np.linalg.norm(self.H, ord = 1)

    def variant_rho(self):
