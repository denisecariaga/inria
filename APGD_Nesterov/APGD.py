import numpy as np
from time import clock
from math import sqrt, inf, isnan
from scipy.sparse import linalg, csr_matrix, csc_matrix

from APGD_Nesterov.Data.read_fclib import *



class Data:
	def Es_matrix(self, vector):
		E_ = np.array([])
		u_per_contact = np.split(vector, self.nc)

		for i in range(int(self.nc)):
			E_ = np.concatenate((E_, np.array([1, 0, 0]) * self.mu[i] * np.linalg.norm(u_per_contact[i][1:])))
		E = E_[:, np.newaxis]

		return np.squeeze(E)

	def __init__(self, problem_data):
		problem = hdf5_file(problem_data)

		self.M = problem.M.tocsc().toarray()
		self.f = problem.f
		self.H = csc_matrix.transpose(problem.H.tocsc()).toarray()
		self.H_T = np.matrix.transpose(self.H)
		self.w = problem.w
		self.W = np.matrix.dot(self.H, np.matrix.dot(np.linalg.inv(self.M), self.H_T))
		self.q = self.w - np.matrix.dot(self.H, np.matrix.dot(np.linalg.inv(self.M), self.f))
		self.mu = problem.mu
		self.g = 10 ** (-6)

		# Dimensions (normal,tangential,tangential)
		self.m = np.shape(self.w)[0]
		self.n = np.shape(self.M)[0]
		self.nc = self.m / 3

		self.s = [1 / np.linalg.norm(self.H, 'fro') * self.Es_matrix(np.ones([self.m, ])) /
		          np.linalg.norm(self.Es_matrix(np.ones([self.m, ])))]
		print(f'm={self.m}')
		print(f's={len(self.s[0])}')
		#################################
		############# SET-UP ############
		#################################

		# Set-up of vectors
		self.v = [np.zeros([self.n, ])]
		self.u = [np.zeros([self.m, ])]
		self.r = [np.zeros([self.m, ])]
		self.res = [] # residual

		self.res_norm = []


##########################################################
#################### RHO CLASSES #########################
##########################################################

class NormalRho:
	def __init__(self, W, M, H):
		self.W = W
		self.M = M
		self.H = H

	def rho(self):
		return 1


class SmallerRho:
	def __init__(self, W, M, H):
		self.W = W
		self.M = M
		self.H = H

	def rho(self):
		return 2 / 3


class WRho:
	def __init__(self, W, M, H):
		self.W = W
		self.M = M
		self.H = H

	def rho(self):
		return 1 / np.linalg.norm(self.W)


class EigenWRho:
	def __init__(self, W, M, H):
		self.W = W
		self.M = M
		self.H = H

	def rho(self):
		eig, eigv = linalg.eigs(self.W)
		eig_max = np.absolute(np.amax(eig))
		return 1 / eig_max


class GhadimiRho:
	def __init__(self, W, M, H):
		self.W = W
		self.M = M
		self.H = H

	def rho(self):
		eig, eig_v = linalg.eigs(self.W)
		eig_max = np.absolute(np.amax(eig))
		eig_min = np.absolute(np.min(eig[np.nonzero(eig)]))
		return 1 / sqrt(eig_max * eig_min)


class DiCairamoRho:
	def __init__(self, W, M, H):
		self.W = W
		self.M = M
		self.H = H

	def rho(self):
		eig, eig_v = linalg.eigs(self.M)
		eig_max = np.absolute(np.amax(eig))
		eig_min = np.absolute(np.min(eig[np.nonzero(eig)]))
		return sqrt(eig_max * eig_min)


class AcaryRho:
	def __init__(self, W, M, H):
		self.W = W
		self.M = M
		self.H = H

	def rho(self):
		return np.linalg.norm(self.M, ord=1) / np.linalg.norm(self.H, ord=1)


##########################################################
#################### APGD METHOD #########################
##########################################################


class APGDMethod:
	def __init__(self, problem_data, rho_class):
		self.r = problem_data.r
		self.W = problem_data.W
		self.q = problem_data.q
		self.M = problem_data.M
		self.H = problem_data.H
		self.m = problem_data.m
		self.nc = problem_data.nc  # m/3
		self.mu = problem_data.mu
		self.dim1 = 3
		self.rho = eval(rho_class)(self.W, self.M, self.H).rho()
		self.g = problem_data.g
		self.v = problem_data.v
		self.u = problem_data.u
		self.s = problem_data.s
		self.res = problem_data.res
		self.res_norm = problem_data.res_norm

	# Formula for the nonsmooth function of the problem
	def Es_matrix(self, vector):
		E_ = np.array([])
		u_per_contact = np.split(vector, self.nc)

		for i in range(int(self.nc)):
			E_ = np.concatenate((E_, np.array([1, 0, 0]) * self.mu[i] * np.linalg.norm(u_per_contact[i][1:])))
		E = E_[:, np.newaxis]

		return np.squeeze(E)

	# Formula to project a vector into the cone
	def project(self, vector):
		vector_per_contact = np.split(vector[0], self.nc)
		projected = np.array([])

		for i in range(int(self.nc)):
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
		if k==1:
			ret = np.zeros([self.m,])
		elif k==2:
			ret = np.zeros([self.m,])
		else:
			ret = self.r[k - 1] + ((k - 2) / (k + 1)) * (self.r[k - 1] - self.r[k - 2])
		return ret

	def update_r(self, k):
		if k==1:
			r = np.zeros([self.m,])
		else:
			r = self.project(
			self.accelerate(k) - self.rho * (np.matrix.dot(self.W, self.accelerate(k)) + (self.q + self.s)))
		self.r.append(r)

	def residual_update(self, k):
		residual = (1 / (self.m * self.g)) * (self.r[k-1] - self.project(
			self.r[k-1] - self.g * (np.matrix.dot(self.W, self.accelerate(k)) + (self.q + self.s))))
		self.res.append(residual)

	def norm_update(self, k):
		self.res_norm.append(np.linalg.norm(self.res[k-1], ord=2))

	# Updating rho usind radio 1
	def update_rho_1(self, k, L, L_min, factor, rho_k_minus_1):
		rho_k = rho_k_minus_1
		vector = self.r[k-1] - rho_k * (np.matrix.dot(self.W, self.r[k-1]) + (self.q + self.s))
		bar_r_k = self.project(vector)

		ratio_k = rho_k * (np.linalg.norm(np.matrix.dot(self.W, self.r[k-1]) - np.matrix.dot(self.W, bar_r_k), ord=2)
		                   * (1 / np.linalg.norm(self.r[k-1] - bar_r_k)))
		while ratio_k > L:
			rho_k = factor * rho_k
			vector = self.r[k-1] - rho_k * (np.matrix.dot(self.r[k-1],self.W) + (self.q + self.s))
			bar_r_k = self.project(vector)

			ratio_k = rho_k * (
					np.linalg.norm(np.matrix.dot(self.W, self.r[k-1]) - np.matrix.dot(self.W, bar_r_k), ord=2)
					* (1 / np.linalg.norm(self.r[k-1] - bar_r_k)))
		if ratio_k < L_min:
			rho_k = (1 / factor) * rho_k
		return rho_k

	# Updating rho using radio 2
	def update_rho_2(self, k, L, L_min, factor, rho_k_minus_1):
		rho_k = rho_k_minus_1
		vector = self.r[k-1] - rho_k * (np.matrix.dot(self.W, self.r[k-1]) + (self.q + self.s))
		bar_r_k = self.project(vector)

		ratio_k = rho_k*(np.matrix.dot((np.matrix.dot(self.r[k-1], self.W) - np.matrix.dot(self.W, bar_r_k)),(self.r[k-1] - bar_r_k)))* ((1 / np.linalg.norm(self.r[k-1] - bar_r_k)) ** 2)
		if not isnan(ratio_k):
			while ratio_k > L:
				rho_k = factor * rho_k
				vector = self.r[k-1] - rho_k * (np.matrix.dot(self.r[k-1], self.W) + (self.q + self.s))
				bar_r_k = self.project(vector)
				ratio_k = rho_k*(np.matrix.dot((np.matrix.dot(self.r[k-1], self.W) - np.matrix.dot(self.W, bar_r_k)),(self.r[k-1] - bar_r_k)))* ((1 / np.linalg.norm(self.r[k-1] - bar_r_k)) ** 2)
		if ratio_k < L_min:
			rho_k = (1 / factor) * rho_k
		return rho_k

	#############################################################
	################### SOLVERS OF THE PROBLEM ##################
	#############################################################

	# Solving the frictional contact problem with fixed rho
	def APGD_N(self, tolerance_r, tolerance_s, iter_max):
		start = clock()
		# This for is to update the value of s
		for j in range(1, iter_max):
			# This while is to solve the problem with s fixed
			k = 1
			error = inf
			while error > tolerance_r and k < iter_max:
				self.update_r(k)
				self.residual_update(k)
				self.norm_update(k)
				error = self.res_norm[k-1]
				k = k+1
			#Updating the value of s
			self.s.append(self.Es_matrix(csr_matrix.dot(self.W, self.r[-1]) + self.q))
			s_per_contact_j1 = np.split(self.s[-1], self.nc)
			s_per_contact_j0 = np.split(self.s[-2], self.nc)
			count = 0
			# Stopping condition of s
			for i in range(int(self.nc)):
				if np.linalg.norm(s_per_contact_j1[i] - s_per_contact_j0[i]) / np.linalg.norm(
						s_per_contact_j0[i]) > tolerance_s:
					count += 1
			if count < 1:
				break
		end = clock()
		return end - start

	# Solving the frictional contact problem with variable rho and ratio 1
	def APGD1_V(self, tolerance_r, tolerance_s, iter_max):
		start = clock()
		list_rho = []
		list_error_s_general_norm = []
		# This for is to update the value of s
		for j in range(1, iter_max):
			# This while is to solve the problem with s fixed
			k = 1
			error = inf
			list_rho.append(self.rho)
			#self.rho = 1   # Fixing the first rho
			while error > tolerance_r and k < iter_max:
				self.rho = self.update_rho_1(k, 0.9, 0.3, 2/3, self.rho)
				self.update_r(k)
				self.residual_update(k)
				self.norm_update(k)
				error = self.res_norm[k-1]
				k += 1
			# Updating the value of s
			self.s.append(self.Es_matrix(csr_matrix.dot(self.W, self.r[-1]) + self.q))
			s_per_contact_j1 = np.split(self.s[-1], self.nc)
			s_per_contact_j0 = np.split(self.s[-2], self.nc)
			list_error_s_it = []
			count = 0
			# Stopping condition of s
			for i in range(int(self.nc)):
				error_s_contact_i = np.linalg.norm(s_per_contact_j1[i] - s_per_contact_j0[i]) / np.linalg.norm(
						s_per_contact_j0[i])
				list_error_s_it.append(error_s_contact_i)
				if  error_s_contact_i > tolerance_s:
					count += 1
			list_error_s_general_norm.append(np.linalg.norm(list_error_s_it))
			if count < 1:
				break
		end = clock()
		timing = end-start
		return [timing, self.s, list_rho, list_error_s_general_norm ]

	# Solving the frictional contact problem with variable rho and ratio 2
	def APGD2_V(self, tolerance_r, tolerance_s, iter_max):
		start = clock()
		# This for is to update the value of s
		for j in range(1, iter_max):
			# This while is to solve the problem with s fixed
			k = 1
			error = inf
			#self.rho = 1  # Fixing the first rho
			while error > tolerance_r and k < iter_max:
				self.rho = self.update_rho_2(k, 0.9, 0.3, 2/3, self.rho)
				self.update_r(k)
				self.residual_update(k)
				self.norm_update(k)
				error = self.res_norm[k-1]
				k += 1
			# Updating the value of s
			self.s.append(self.Es_matrix(csr_matrix.dot(self.W, self.r[-1]) + self.q))
			s_per_contact_j1 = np.split(self.s[-1], self.nc)
			s_per_contact_j0 = np.split(self.s[-2], self.nc)
			count = 0
			# Stopping condition of s
			for i in range(int(self.nc)):
				if np.linalg.norm(s_per_contact_j1[i] - s_per_contact_j0[i]) / np.linalg.norm(
						s_per_contact_j0[i]) > tolerance_s:
					count += 1
			if count < 1:
				break

		end = clock()
		return end - start
