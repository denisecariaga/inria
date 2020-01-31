#########################################################
###################### IMPORT DATA ######################
#########################################################
# Import librearies
import numpy as np
import pickle

dict_master = pickle.load(open("time_solver.p", "rb"))

#########################################################
######################### CODE ##########################
#########################################################



# Definition of list
all_solvers = ['cp_N', 'cp_R', 'cp_RR', 'vp_N_He', 'vp_R_He', 'vp_RR_He', 'vp_N_Spectral', 'vp_R_Spectral',
               'vp_RR_Spectral', 'vp_N_Wohlberg', 'vp_R_Wohlberg', 'vp_RR_Wohlberg']
rho_optimal_time = ['acary (time)', 'dicairano (time)', 'ghadimi (time)', 'normal (time)']

# Ratio problem/solver
for each_problem_data in dict_master:
    for each_rho_time in rho_optimal_time:
        timing_rho = []
        for each_solver in range(len(all_solvers)):
            timing_rho.append(each_problem_data[each_solver][each_rho_time])
        timing_rho_array = np.asarray(timing_rho)
        timing_rho_ratio = timing_rho_array / np.nanmin(timing_rho_array)
        for each_solver in range(len(all_solvers)):
            each_problem_data[each_solver]['p_ratio ' + each_rho_time] = timing_rho_ratio[each_solver]

# Save the data
pickle.dump(dict_master, open("ratio_solver.p", "wb"))
