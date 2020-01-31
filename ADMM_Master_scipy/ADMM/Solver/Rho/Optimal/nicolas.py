import numpy as np


def nicolas(A, M, A_T):
    M_norm1 = np.linalg.norm(M, 1)
    A_norm1 = np.linalg.norm(A, 1)

    optimal = (np.sqrt(M_norm1) / A_norm1)

    return optimal
