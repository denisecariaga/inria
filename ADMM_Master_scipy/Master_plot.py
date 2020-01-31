#########################################################
###################### IMPORT DATA ######################
#########################################################
# Import librearies
import numpy as np
import matplotlib.pyplot as plt

import pickle

performance_profile = pickle.load(open("performance_profile.p", "rb"))

#########################################################
######################### CODE ##########################
#########################################################


# Definition of list
color = ['#9ACD32', '#FFFF00', '#40E0D0', '#FF6347', '#A0522D', '#FF00FF', '#FFA500', '#808000', '#000080', '#006400',
         '#0000FF', '#000000']
# ['yellowgreen','yellow','violet','turquoise','tomato','magenta','salmon','orange','olive','navy','darkgreen','blue','black']
tau_ratio = np.arange(1.0, 11.0, 0.01)
all_solvers = ['cp_N', 'cp_R', 'cp_RR', 'vp_N_He', 'vp_R_He', 'vp_RR_He', 'vp_N_Spectral', 'vp_R_Spectral',
               'vp_RR_Spectral', 'vp_N_Wohlberg', 'vp_R_Wohlberg', 'vp_RR_Wohlberg']
rho_optimal = ['acary', 'dicairano', 'ghadimi', 'normal']

# Plot
for each_rho_time in range(len(rho_optimal)):
    for s in range(len(all_solvers)):
        plt.plot(tau_ratio, performance_profile[s][each_rho_time], color[s], label=all_solvers[s])
        plt.hold(True)
    plt.ylabel('Performance')
    plt.xlabel('Tau')
    plt.title('Performance profile for ' + rho_optimal[each_rho_time])
    plt.legend()
    plt.show()
