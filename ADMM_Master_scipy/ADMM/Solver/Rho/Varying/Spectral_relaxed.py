#################################
## Spectral parameter - update ##
#################################
import numpy as np
from scipy.sparse import csr_matrix

eps_corr = 0.2
mod = 2  # varying every 'mod' iterations


def penalty(A, Av, u, w, b, xi_hat, ratio, rG, xiG, rho, v, k):
    if k % mod == 0:
        step = -k * (mod - 1) / mod

        # Set up of needed constants
        rG.append(Av - u[k] + w + b)  # rG[k+1]
        xiG.append(ratio * (xi_hat[k] + rG[k + 1 + step]))  # xiG[k+1]

        # Set up of new variables
        Dlambda = rho[k] * xiG[k + 1 + step] - rho[k + step] * xiG[k + step]
        DH = csr_matrix.dot(A, v[k + 1] - np.squeeze(v[k]))
        DG = - u[k + 1] + u[k]

        # Definitions of inner products
        Dlambda_dot = np.dot(np.transpose(Dlambda), Dlambda)
        DH_dot = np.dot(np.transpose(DH), DH)
        DG_dot = np.dot(np.transpose(DG), DG)

        DH_Dlambda_dot = np.dot(np.transpose(DH), Dlambda)
        DG_Dlambda_dot = np.dot(np.transpose(DG), Dlambda)

        # Definitions of norms
        Dlambda_norm = np.linalg.norm(Dlambda)
        DH_norm = np.linalg.norm(DH)
        DG_norm = np.linalg.norm(DG)

        # Definition of alfa and beta SD/MG
        alfa_SD = Dlambda_dot / DH_Dlambda_dot
        alfa_MG = DH_Dlambda_dot / DH_dot

        beta_SD = Dlambda_dot / DG_Dlambda_dot
        beta_MG = DG_Dlambda_dot / DG_dot

        # Election of alfa and beta hat
        if 2.0 * alfa_MG > alfa_SD:
            alfa_hat = alfa_MG
        else:
            alfa_hat = alfa_SD - alfa_MG / 2.0

        if 2.0 * beta_MG > beta_SD:
            beta_hat = beta_MG
        else:
            beta_hat = beta_SD - beta_MG / 2.0

        # Correlations
        alfa_corr = DH_Dlambda_dot / (DH_norm * Dlambda_norm)
        beta_corr = DG_Dlambda_dot / (DG_norm * Dlambda_norm)

        # Penalty parameter update
        if alfa_corr > eps_corr and beta_corr > eps_corr:
            rhos = np.sqrt(alfa_hat * beta_hat)
        elif alfa_corr > eps_corr and beta_corr <= eps_corr:
            rhos = alfa_hat
        elif alfa_corr <= eps_corr and beta_corr > eps_corr:
            rhos = beta_hat
        else:
            rhos = rho[k]

        return rhos
    else:
        return rho[k]
