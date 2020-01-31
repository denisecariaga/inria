# Penalty parameter
def penalty(rho, r_norm, s_norm):
    mu = 10.0
    factor = 2.0

    if r_norm > mu * s_norm:
        return rho * factor
    elif s_norm > mu * r_norm:
        return rho / factor
    else:
        return rho
