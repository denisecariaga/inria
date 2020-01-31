import numpy as np
from scipy.sparse import linalg


def dicairano(A, M, A_T):
    eig, eig_vec = linalg.eigs(M)

    eigmax = np.absolute(np.amax(eig))
    eigmin = np.absolute(np.min(eig[np.nonzero(eig)]))
    return np.sqrt(eigmax * eigmin)
