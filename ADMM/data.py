import numpy as np


# n = 5
# n_c = 2
# m = 3*n_c = 6

class ProblemData:
    def __init__(self):
        # m sized variables
        self.r = [np.zeros((6,))]
        self.p = [np.zeros((6,))]
        self.s = [np.zeros((6,))]

        # mxn sized matrix
        self.H = np.ones((6, 5))

        # nxn sized matrix
        self.M = [np.ones((5, 5))]

        # n sized parameters
        self.f = [np.ones((5,))
                  ]
        # mxm sized matrix
        self.Id = [np.identity(6)]
        self.W = (self.H.dot(np.linalg.inv(self.M))).dot(self.H.transpose())

        # m sized parameters
        self.w = np.ones((6,))
        self.q = - (self.H.dot(np.linalg.inv(self.M))).dot(self.f)
        self.u = self.W.dot(self.r) + self.q
        self.u_hat = self.u +self.s

        # n_c sized parameters
        self.mu = np.random.uniform(0, 1, 2)






p = ProblemData()
print(p.f[0][3])
print(np.shape(p.w)[0])
