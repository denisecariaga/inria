from APGD_Nesterov import APGD
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from APGD_Nesterov.Data.read_fclib import *

NUM_ITER = 1000

################## IMPORTING PROBLEMS ######################
dir = "/Users/denisecarolinacariagasandoval/inria/APGD_Nesterov/Data/box_stacks"
problems = os.listdir(dir)
problems.sort()
print('etapa0')
##################### IMPORTING RHO ########################

rhos = ['NormalRho', 'SmallerRho', 'WRho', 'EigenWRho', 'GhadimiRho', 'DiCairamoRho', 'AcaryRho']

################### IMPORTING APGD METHOD for each rho ##################
list_master = []

for problem in problems:
	print('etapa0.4')
	pr = hdf5_file(problem)
	print('open problem')
	##################### IMPORTING DATA ######################
	data = APGD.Data(problem)
	print(type(data))
	print('etapa0.5')
	dict_problem = {'problem': problem, 'solver': 'APGD_rho_fixed'}
	for rho in rhos:
		timing = APGD.APGDMethod(data, rho).APGD_N(10**(-3),10**(-3),NUM_ITER)
		dict_problem[rho+'(time)'] = timing
	list_master.append(dict_problem)

# Save the data
pickle.dump(list_master, open('time_solver.p', 'wb'))

print('etapa1')
#########################################################
######################### CODE ##########################
#########################################################


#Definition of list
rho_optimal_time = ['NormalRho(time)', 'SmallerRho(time)', 'WRho(time)', 'EigenWRho(time)', 'GhadimiRho(time)',
                    'DiCairamoRho(time)', 'AcaryRho(time)']

#Ratio problem/solver
for each_problem_data in list_master:
	for each_rho_time in rho_optimal_time:
		timing_rho = []
		timing_rho.append(each_problem_data[each_rho_time])
		timing_rho_array = np.asarray(timing_rho)
		timing_rho_ratio = timing_rho_array / np.nanmin(timing_rho_array)
		each_problem_data['p_ratio_' + each_rho_time] = timing_rho_ratio[0]

#Save the data
pickle.dump(list_master, open("ratio_solver.p", "wb"))

print('etapa2')

#########################################################
######################### CODE ##########################
#########################################################

#Definition of list
rho_optimal_ratio = ['p_ratio_NormalRho(time)', 'p_ratio_SmallerRho(time)', 'p_ratio_WRho(time)',
                     'p_ratio_EigenWRho(time), p_ratio_GhadimiRho(time), p_ratio_DiCairamoRho(time)',
                     'p_ratio_AcaryRho(time)']
tau_ratio = np.arange(1.0,11.0,0.01)

#Performance problem/solver
performance_general = []

performance_solver = []
for each_rho_ratio in rho_optimal_ratio:
	performance_rho = []
	for tau in tau_ratio:
		cardinal_number = 0.0
		for each_problem_data in list_master:
			if each_problem_data[each_rho_ratio] <= tau:
				cardinal_number += 1.0
		performance_tau = cardinal_number / len(list_master)
		performance_rho.append(performance_tau)
	performance_solver.append(performance_rho)
performance_general.append(performance_solver)

#Save the data
pickle.dump(performance_general, open("performance_profile.p", "wb"))
print('etapa3')

color = ['#9ACD32','#FFFF00','#40E0D0','#FF6347','#A0522D','#FA8072','#FFA500','#808000','#000080','#006400','#0000FF','#000000']
#['yellowgreen','yellow','violet','turquoise','tomato','sienna','salmon','orange','olive','navy','darkgreen','blue','black']
tau_ratio = np.arange(1.0,11.0,0.01)

#Plot
for each_rho_time in range(len(rhos)):
	plt.plot(tau_ratio, performance_general[each_rho_time], color[each_rho_time], label = '')
	plt.hold(True)
	plt.ylabel('Performance')
	plt.xlabel('Tau')
	plt.title('Performance profiles for '+rhos[each_rho_time])
	plt.legend()
	plt.show()
