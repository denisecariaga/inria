'''
% Solves the following problem via ADMM:
%
%   minimize     (1/2)*v'*M*v + f'*v + indicator(u)
%   subject to   u = Av + b
'''
######################
## IMPORT LIBRARIES ##
######################

# Math libraries
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import linalg

# Timing
import time
import pickle
# Import data
from Data.read_fclib import *

# Plot residuals
from Solver.ADMM_iteration.Numerics.plot import *

# Initial penalty parameter
import Solver.Rho.Optimal

# Max iterations and kind of tolerance
from Solver.Tolerance.iter_totaltolerance import *

# b = Es matrix
from Data.Es_matrix import *

# Projection onto second order cone
from Solver.ADMM_iteration.Numerics.projection import *


def cp_N(problem_data, rho_method):
    ##################################
    ############# REQUIRE ############
    ##################################

    start = time.clock()
    problem = hdf5_file(problem_data)

    M = problem.M.tocsc()
    f = problem.f
    A = csc_matrix.transpose(problem.H.tocsc())
    A_T = csr_matrix.transpose(A)
    w = problem.w
    mu = problem.mu

    # Dimensions (normal,tangential,tangential)
    dim1 = 3
    dim2 = np.shape(w)[0]

    # Problem size
    n = np.shape(M)[0]
    p = np.shape(w)[0]


    b = 0.1 * Es_matrix(w, mu, np.ones([p, ])) / np.linalg.norm(Es_matrix(w, mu, np.ones([p, ])))

    #################################
    ############# SET-UP ############
    #################################

    # Set-up of vectors
    v = [np.zeros([n, ])]
    u = [np.zeros([p, ])]  # this is u tilde, but in the notation of the paper is used as hat [np.zeros([10,0])]
    xi = [np.zeros([p, ])]
    r = [np.zeros([p, ])]  # primal residual
    s = [np.zeros([p, ])]  # dual residual
    r_norm = [0]
    s_norm = [0]
    e = []  # restart

    # Optimal penalty parameter
    start = time.clock()
    rho_string = 'Solver.Rho.Optimal.' + rho_method + '(A,M,A_T)'
    rho = eval(rho_string)

    ########################################
    ## TERMS COMMON TO ALL THE ITERATIONS ##
    ########################################

    # Super LU factorization of M + rho * dot(M_T,M)
    P = M + rho * csc_matrix.dot(A_T, A)
    LU = linalg.splu(P)

    ################
    ## ITERATIONS ##
    ################
    for k in range(MAXITER):

        ################
        ## v - update ##
        ################
        RHS = -f + rho * csc_matrix.dot(A_T, -w - b - xi[k] + u[k])
        v.append(LU.solve(RHS))  # v[k+1]

        ################
        ## u - update ##
        ################
        Av = csr_matrix.dot(A, v[k + 1])
        vector = Av + xi[k] + w + b
        u.append(projection(vector, mu, dim1, dim2))  # u[k+1]

        ########################
        ## residuals - update ##
        ########################
        s.append(rho * csc_matrix.dot(A_T, (u[k + 1] - u[k])))  # s[k+1]
        r.append(Av - u[k + 1] + w + b)  # r[k+1]

        #################
        ## xi - update ##
        #################
        xi.append(xi[k] + r[k + 1])  # xi[k+1]

        ####################
        ## stop criterion ##
        ####################
        pri_evalf = np.amax(
            np.array([np.linalg.norm(csr_matrix.dot(A, v[k + 1])), np.linalg.norm(u[k + 1]), np.linalg.norm(w + b)]))
        eps_pri = np.sqrt(p) * ABSTOL + RELTOL * pri_evalf

        dual_evalf = np.linalg.norm(rho * csc_matrix.dot(A_T, xi[k + 1]))
        eps_dual = np.sqrt(n) * ABSTOL + RELTOL * dual_evalf

        r_norm.append(np.linalg.norm(r[k + 1]))
        s_norm.append(np.linalg.norm(s[k + 1]))
        if r_norm[k + 1] <= eps_pri and s_norm[k + 1] <= eps_dual:
            break

    # end rutine

    end = time.clock()
    ####################
    ## REPORTING DATA ##
    ####################
    # plotit(r,s,start,end,'Without acceleration / Without restarting for '+problem_data+' for rho: '+rho_method)

    time_tot = end - start
    return time_tot

