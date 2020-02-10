import numpy as np
from scipy.sparse import linalg
from scipy.sparse import csc_matrix
'''
print([1 for i in range(3)])


def master(solver, problem, rho_method):
    STRING = 'ADMM.' + solver + '("' + problem + '", "' + rho_method + '")'
    return eval(STRING)


print(master('solver', 'problem', 'rho'))
'''

M = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
# print(M)

M_new = csc_matrix(M)
print(M_new[0])

LU = linalg.splu(M_new)
b = np.array([1, 2, 3])
print(np.shape(np.array([1, 2, 3])))
print(LU.solve(np.array([1, 2, 3])))

H = np.ones((6, 5))
print(H)
print(H.transpose())
print(type(H))