from APGD_Nesterov import APGD
import os

NUM_ITER = 1000



################## IMPORTING PROBLEMS ######################

problems = os.listdir("ADMM/Data/box_stacks/")
problems.sort()

##################### IMPORTIND RHO ########################
Rho = APGD.Rho(W, M, H, tau, d)

rhos = [Rho.normal_rho(), Rho.smaller_rho(), Rho.w_rho(), Rho.eigen_w_rho(), Rho.ghadimi_rho(), Rho.di_cairamo_rho(),
        Rho.acary_rho()]

################### IMPORTING APGD METHOD for each rho ##################
for problem in problems:
	##################### IMPORTING DATA ######################
	data = APGD.Data()

	r = data.r
	W = data.W
	H = data.H
	q = data.q
	M = data.M
	s = data.s
	f = data.f
	m = data.m
	n = data.n
	nc = data.nc

	tau = 0
	d = 0
	for rho in rhos:
		Method = APGD.APGDMethod()
