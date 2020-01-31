######################################
## accelerated ADMM without restart ##
######################################

import numpy as np


def minusr(tau, u, u_hat, xi, xi_hat, k, ratio):
    tau.append(0.5 * (1 + np.sqrt(1 + 4 * np.square(tau[k]))))  # tau[k+1]
    alpha = (tau[k] - 1) / tau[k + 1]
    u_hat.append(u[k + 1] + alpha * (u[k + 1] - u[k]))  # u_hat[k+1]
    xi_hat.append(xi[k + 1] + alpha * (xi[k + 1] - ratio * xi[k]))  # xi_hat[k+1]
