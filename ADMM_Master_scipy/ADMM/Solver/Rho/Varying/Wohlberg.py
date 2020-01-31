import numpy as np


def penalty(rho, r_norm, s_norm, mu):
    mu = 10.0
    factormax = 100.0

    ratiosqrt = np.sqrt(r_norm / s_norm)
    if 1.0 <= ratiosqrt < factormax:
        factor = ratiosqrt
    elif 1.0 / factormax < ratiosqrt < 1.0:
        factor = 1.0 / ratiosqrt
    else:
        factor = factormax

    if r_norm > mu * s_norm:
        rhos = rho * factor
    elif s_norm > mu * r_norm:
        rhos = rho / factor
    else:
        rhos = rho

    return rhos
