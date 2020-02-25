#########################################################
#################### MASTER FUNCTION ####################
#########################################################
# Import librearies
import numpy as np
import pickle
import os
import ADMM


def master(solver, problem, rho_method):
    STRING = 'ADMM.' + solver + '("' + problem + '", "' + rho_method + '")'
    return eval(STRING)


#########################################################
###################### IMPORT DATA ######################
#########################################################

# Import all the problems hdf5


all_problems = os.listdir("ADMM/Data/box_stacks/")
all_problems.sort()

# Import all the solvers
all_solvers = ['cp_N', 'cp_R', 'cp_RR', 'vp_N_He', 'vp_R_He', 'vp_RR_He', 'vp_N_Spectral', 'vp_R_Spectral',
               'vp_RR_Spectral', 'vp_N_Wohlberg', 'vp_R_Wohlberg', 'vp_RR_Wohlberg']

#########################################################
######################### CODE ##########################
#########################################################


# Definition of list
list_master = []
rho_optimal = ['acary', 'dicairano', 'ghadimi', 'normal']

# Time problem/solver
for each_problem in all_problems:
    print('---' + each_problem + '---')
    list_problem = []
    for each_solver in all_solvers:
        dict_solver = {'problem': each_problem, 'solver': each_solver}
        for each_rho in rho_optimal:
            #print(each_solver + ': ' + each_rho)
            # timing = master(each_solver, each_problem, each_rho)
            try:
                timing = master(each_solver, each_problem, each_rho)
            except:
                timing = np.nan  # NaN
            print(timing)
            dict_solver[each_rho + ' (time)'] = timing
        list_problem.append(dict_solver)
    list_master.append(list_problem)

# Save the data
pickle.dump(list_master, open("time_solver.p", "wb"))
