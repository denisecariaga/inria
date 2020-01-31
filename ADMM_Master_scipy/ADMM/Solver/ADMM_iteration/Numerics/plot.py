import numpy as np
import matplotlib.pyplot as plt


def plotit(r, s, start, end, title):
    R = [np.linalg.norm(k) for k in r]
    S = [np.linalg.norm(k) for k in s]
    plt.semilogy(R, label='||r||')
    plt.hold(True)
    plt.semilogy(S, label='||s||')
    plt.hold(True)
    plt.ylabel('Residuals')
    plt.xlabel('Iteration')
    plt.text(len(r) / 2, np.log(np.amax(S) + np.amax(R)) / 10, 'N_iter = ' + str(len(r) - 1))
    plt.text(len(r) / 2, np.log(np.amax(S) + np.amax(R)) / 100, 'Total time = ' + str((end - start) * 10 ** 3) + ' ms')
    plt.text(len(r) / 2, np.log(np.amax(S) + np.amax(R)) / 1000,
             'Time_per_iter = ' + str(((end - start) / (len(r) - 1)) * 10 ** 3) + ' ms')
    plt.title(title)
    plt.legend()
    plt.show()
