#########################################################
###################### IMPORT DATA ######################
#########################################################

import pickle
import sys



print('ratio_solver.p')
dict_master = pickle.load(open("ratio_solver.p", "rb"))

#########################################################
######################### CODE ##########################
#########################################################

#Import librearies
import numpy as np

#Definition of list
all_solvers = ['cp_N', 'cp_R', 'cp_RR', 'vp_N_He', 'vp_R_He', 'vp_RR_He', 'vp_N_Spectral', 'vp_R_Spectral', 'vp_RR_Spectral', 'vp_N_Wohlberg', 'vp_R_Wohlberg', 'vp_RR_Wohlberg']
rho_optimal_ratio = ['p_ratio acary (time)', 'p_ratio dicairano (time)', 'p_ratio ghadimi (time)', 'p_ratio normal (time)']
tau_ratio = np.arange(1.0, 11.0, 0.01)

#Performance problem/solver
performance_general = []
for each_solver in range(len(all_solvers)):
	performance_solver = []
	for each_rho_ratio in rho_optimal_ratio:
		performance_rho = []
		for tau in tau_ratio:
			cardinal_number = 0.0
			for each_problem_data in dict_master:
				if each_problem_data[each_solver][each_rho_ratio] <= tau:
					cardinal_number += 1.0
			performance_tau = cardinal_number / len(dict_master)
			performance_rho.append(performance_tau)
		performance_solver.append(performance_rho)
	performance_general.append(performance_solver)

#Save the data	
pickle.dump(performance_general, open("performance_profile.p", "wb"))
