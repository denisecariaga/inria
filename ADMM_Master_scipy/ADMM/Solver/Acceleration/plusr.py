######################################
## accelerated ADMM with restart ##
######################################

import numpy as np

eta = 0.999
def plusr(tau,u,u_hat,xi,xi_hat,k,e,rho):
	e.append(rho * np.square(np.linalg.norm(xi[k+1]-xi_hat[k])) + rho * np.square(np.linalg.norm(u[k+1]-u_hat[k]))) #e[k]

	if e[k] < eta * e[k-1]:
		tau.append(0.5 * (1 + np.sqrt(1 + 4 * np.square(tau[k])))) #tau[k+1]
		alpha = (tau[k] - 1) / tau[k+1]
		u_hat.append(u[k+1] + alpha * (u[k+1] - u[k])) #u_hat[k+1]
		xi_hat.append(xi[k+1] + alpha * (xi[k+1] - xi[k])) #xi_hat[k+1]
	else:
		tau.append(1) #tau[k+1]
		u_hat.append(u[k]) #u_hat[k+1]
		xi_hat.append(xi[k]) #xi_hat[k+1]
		e[k] = e[k-1] / eta
