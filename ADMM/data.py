import numpy as np


# n = 5
# n_c = 2
# m = 3*n_c = 6

class ProblemData:
    def __init__(self):
        # n sized parameters
        self.M = [np.ones((5, 5))]
        self.f = [np.ones((5,))]
        # m sized parameters
        self.Id = [np.identity(6)]
        self.w = np.ones((6,))
        self.H = np.ones((6, 5))


p = ProblemData()
print(p.f[0][3])
print(np.shape(p.w)[0])
