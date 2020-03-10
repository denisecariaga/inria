from APGD_Nesterov import APGD
import os
import pickle

NUM_ITER = 1000



################## IMPORTING PROBLEMS ######################

problems = os.listdir("ADMM_Master_scipy/ADMM/Data/box_stacks/")
problems.sort()

##################### IMPORTING RHO ########################

rhos = ['NormalRho', 'SmallerRho', 'WRho', 'EigenWRho', 'GhadimiRho', 'DiCairamoRho', 'AcaryRho']

################### IMPORTING APGD METHOD for each rho ##################
list_master = []

for problem in problems:
	##################### IMPORTING DATA ######################
	data = APGD.Data(problem)

	dict_solver = {'problem': problem, 'solver': 'APGD_rho_fijo'}
	for rho in rhos:
		timing = APGD.APGDMethod(data, rho)
		dict_solver[rho+'(time)'] = timing
	list_master.append(dict_solver)

# Save the data
pickle.dump(list_master, open('time_solver.p', 'wb'))



