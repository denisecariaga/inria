import numpy as np


def Es_matrix(w, mu, u):
    dim1 = 3
    dim2 = w.shape[0]
    E_ = np.array([])

    u_per_contact = np.split(u, dim2 / dim1)

    for i in range(dim2 / dim1):
        E_ = np.concatenate((E_, np.array([1, 0, 0]) * mu[i] * np.linalg.norm(u_per_contact[i][1:])))
    E = E_[:, np.newaxis]

    return np.squeeze(E)
