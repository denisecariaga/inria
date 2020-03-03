import numpy as np
from time import time
from math import sqrt
from scipy.sparse import linalg, csr_matrix, csc_matrix

from ADMM import Es_matrix
from ADMM.Data.read_fclib import *
import random


class Data:
	def Es_matrix(self, u):
		E_ = np.array([])
		u_per_contact = np.split(u, self.nc)

		for i in range(int(self.nc)):
			E_ = np.concatenate((E_, np.array([1, 0, 0]) * self.mu[i] * np.linalg.norm(u_per_contact[i][1:])))
		E = E_[:, np.newaxis]

		return np.squeeze(E)

	def __init__(self, problem_data):
		problem = hdf5_file(problem_data)

		self.M = problem.M.tocsc()
		self.f = problem.f
		self.H = csr_matrix.transpose(problem.H.tocsc())
		self.H_T = csr_matrix.transpose(csr_matrix(self.H))
		self.w = problem.w
		self.W = csr_matrix.multi_dot(self.H_T, np.linalg.inv(self.M), self.H)
		self.q = self.w - csr_matrix.multi_dot(self.H_T, np.linalg.inv(self.M), self.f)
		self.mu = problem.mu
		self.g = 10**(-6)


		# Dimensions (normal,tangential,tangential)
		self.m = np.shape(self.w)[0]
		self.n = np.shape(self.M)[0]
		self.nc = self.n / 3

		self.s = 0.1 * (self.Es_matrix(np.ones([self.m, ])) / np.linalg.norm(self.Es_matrix(np.ones([self.m, ]))))

		#################################
		############# SET-UP ############
		#################################

		# Set-up of vectors
		v = [np.zeros([self.n, ])]
		self.r = [np.zeros([self.m, ])]  # this is u tilde, but in the notation of the paper is used as hat [np.zeros([10,0])]
		xi = [np.zeros([self.m, ])]
		self.res = [np.zeros([self.m, ])]  #  residual

		self.res_norm = [0]


class Rho:
	def __init__(self, W, M, H):
		self.W = W
		self.M = M
		self.H = H

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


class APGDMethod:
	def __init__(self, problem_data, rho_method):
		self.r = problem_data.r
		self.W = problem_data.W
		self.q = problem_data.q
		self.M = problem_data.M
		self.H = problem_data.H
		self.m = problem_data.m
		self.n_c = problem_data.n_c  # m/3
		self.mu = problem_data.mu
		self.dim1 = 3
		self.rho = Rho(self.W, self.M, self.H).rho_method()
		self.g = problem_data.g

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
		return projected

	def accelerate(self, k):
		return self.r[k - 1] + ((k - 2) / (k + 1)) * (self.r[k - 1] - self.r[k - 2])

	def update_r(self, k):
		self.r[k].append(self.project(
			self.accelerate(k) - self.rho * (csr_matrix.dot(self.W, self.accelerate(k)) + self.q)))

	def stop_criteria(self, k, epsilon):
		res = (1 / (self.m * self.g)) * (self.r[k] - self.project(
			self.r[k] - self.g * (csr_matrix.dot(self.W, self.accelerate(k)) + self.q)))
		norm_res = np.linalg.norm(res.toarray(), ord=2)
		if norm_res < epsilon:
			return True
		else:
			return False

	def update_rho_1(self, k, L, L_min, factor, rho_k_minus_1):
		rho_k = rho_k_minus_1
		vector = self.r[k] - rho_k * (csr_matrix.dot(self.W, self.r[k]) + self.q)
		bar_r_k = self.project(vector)

		ratio_k = rho_k * (np.linalg.norm(csr_matrix.dot(self.W, self.r[k]) - csr_matrix.dot(self.W, bar_r_k), ord=2)
		                   * (1 / np.linalg.norm(self.r[k] - bar_r_k)))
		while ratio_k > L:
			rho_k = factor * rho_k
			vector = self.r[k] - rho_k * (csr_matrix.dot(self.W, self.r[k]) + self.q)
			bar_r_k = self.project(vector)

			ratio_k = rho_k * (
					np.linalg.norm(csr_matrix.dot(self.W, self.r[k]) - csr_matrix.dot(self.W, bar_r_k), ord=2)
					* (1 / np.linalg.norm(self.r[k] - bar_r_k)))
		if ratio_k < L_min:
			rho_k = (1 / factor) * rho_k
		return rho_k

	def update_rho_2(self, k, L, L_min, factor, rho_k_minus_1):
		rho_k = rho_k_minus_1
		vector = self.r[k] - rho_k * (csr_matrix.dot(self.W, self.r[k]) + self.q)
		bar_r_k = self.project(vector)

		ratio_k = rho_k * (np.transpose(self.r[k] - bar_r_k)
		                   * (csr_matrix.dot(self.W, self.r[k]) - csr_matrix.dot(self.W, bar_r_k))
		                   * ((1 / np.linalg.norm(self.r[k] - bar_r_k)) ** 2))
		while ratio_k > L:
			rho_k = factor * rho_k
			vector = self.r[k] - rho_k * (csr_matrix.dot(self.W, self.r[k]) + self.q)
			bar_r_k = self.project(vector)

			ratio_k = rho_k * (np.transpose(self.r[k] - bar_r_k)
			                   * (csr_matrix.dot(self.W, self.r[k]) - csr_matrix.dot(self.W, bar_r_k))
			                   * ((1 / np.linalg.norm(self.r[k] - bar_r_k)) ** 2))
		if ratio_k < L_min:
			rho_k = (1 / factor) * rho_k
		return rho_k
