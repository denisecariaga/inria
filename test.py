print([1 for i in range(3)])
def master(solver, problem, rho_method):
    STRING = 'ADMM.' + solver + '("' + problem + '", "' + rho_method + '")'
    return eval(STRING)
print(master('solver', 'problem', 'rho'))