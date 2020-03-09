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


lista = ['a', 'b', 'c']

for i in range(len(lista)):
    print(i)

print('ejemplo clases')

class Example:
    def __init__(self, a):
        self.a = int(a)

    def funcion1(self, k):
        return self.a * k

    def funcion2(self, k):
        return self.a ** k


class rho1:
    def __init__(self, a):
        self.a = int(a)

    def rho(self, k):
        return self.a * k


class rho2:
    def __init__(self, a):
        self.a = a

    def rho(self, k):
        return self.a ** k


class rho3:
    def __init__(self, a):
        self.a = a

    def rho(self, k):
        return self.a + k


class Ejecutar2:
	def __init__(self, clase, a, k):
		self.funcion = clase(a).rho(k)


lista = [rho1, rho2, rho3]
for elem in lista:
	print(Ejecutar2(elem, 2, 3).funcion)
print(Ejecutar2(rho2, 2, 3).funcion)
print('done')
