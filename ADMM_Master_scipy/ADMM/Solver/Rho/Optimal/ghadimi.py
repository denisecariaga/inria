import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import linalg


def ghadimi(A, M, A_T):
    DUAL = csr_matrix.dot(csr_matrix.dot(A, linalg.inv(M)), A_T)
    eig, eig_vect = linalg.eigs(DUAL)  # sparse

    eigmax = np.absolute(np.amax(eig))
    eigmin = np.absolute(np.min(eig[np.nonzero(eig)]))
    return 1 / np.sqrt(eigmax * eigmin)
