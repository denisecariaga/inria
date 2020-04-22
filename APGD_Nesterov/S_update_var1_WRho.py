from APGD_Nesterov import APGD
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from APGD_Nesterov.Data.read_fclib import *
import time

NUM_ITER = 200
inicio = time.clock()
################## IMPORTING PROBLEMS ######################
dir = "/Users/denisecarolinacariagasandoval/inria/APGD_Nesterov/Data/box_stacks"
problems = os.listdir(dir)
problems.sort()
##################### IMPORTING RHO ########################

rhos = ['WRho']

################### IMPORTING APGD METHOD for each rho ##################
list_master = []
list_s = []
list_rho = []
lis_norm_error_s = []
for problem in problems:
	if problem=='Box_Stacks-i0024-549-101.hdf5':
		pr = hdf5_file(problem)
		##################### IMPORTING DATA ######################
		data = APGD.Data(problem)
		for rho in rhos:
			timing = APGD.APGDMethod(data, rho).APGD1_V(10**(-3),10**(-3),NUM_ITER)
			list_s.append(timing[1])
			list_rho.append(timing[2])
			lis_norm_error_s.append(timing[3])

list_norm_s = []
for elem in list_s:
	list_norm_s.append(np.linalg.norm(elem))

print(f'largo de s: {(list_rho)}')

fin = time.clock()

color = ['#9ACD32','#40E0D0','#A0522D','#FA8072','#808000','#000080','#006400','#000000']
#['yellowgreen','violet','turquoise','tomato','sienna','orange','olive','navy','darkgreen','black']


print()

#Plot
for rho in range(1):
	plt.plot(tau_ratio, performance_general[0][rho], color[rho], label = rhos[rho])
	plt.hold(True)
plt.ylabel('Performance')
plt.xlabel('Tau')
plt.title('Performance profiles for APGD  Updating rule for Rho 1')
plt.legend()
plt.show()
