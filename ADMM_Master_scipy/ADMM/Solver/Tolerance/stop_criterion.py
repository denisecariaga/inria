import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix


def stopcriterion(A, A_T, v, u, b, xi, r, s, r_norm, s_norm, p, n, ABSTOL, RELTOL, rho, k):
    pri_evalf = np.array([np.linalg.norm(csr_matrix.dot(A, v[k + 1])), np.linalg.norm(u[k + 1]), np.linalg.norm(b)])
    eps_pri = np.sqrt(p) * ABSTOL + RELTOL * np.amax(pri_evalf)

    dual_evalf = rho * csc_matrix.dot(A_T, xi[k + 1])
    eps_dual = np.sqrt(n) * ABSTOL + RELTOL * np.linalg.norm(dual_evalf)

    r_norm.append(np.linalg.norm(r[k + 1]))
    s_norm.append(np.linalg.norm(s[k + 1]))
    if r_norm[k + 1] <= eps_pri and s_norm[k + 1] <= eps_dual:
        return 'break'
