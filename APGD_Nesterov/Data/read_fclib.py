import h5py
import numpy


def hdf5_file(problem_data):
    filename = '/Users/denisecarolinacariagasandoval/inria/APGD_Nesterov/Data/box_stacks/' + problem_data  # LMGC_GFC3D_CubeH8

    # filename = os.getcwd()+'/Data/Box_Stacks-i0024-642-53.hdf5' #Box_Stacks-i0024-642-53
    # filename = os.getcwd()+'/Data/Box_Stacks-i0029-31-0.hdf5' #Box_Stacks-i0029-31-0
    # filename = os.getcwd()+'/Data/Box_Stacks-i0045-52-1.hdf5' #Box_Stacks-i0045-52-1

    def is_fclib_file(filename):
        r = False
        try:
            with h5py.File(filename, 'r') as f:
                r = 'fclib_local' in f or 'fclib_global' in f
            # except Exception as e:
            #    print(e)
        except:
            pass
        return r

    if (not is_fclib_file(filename)):
        exit()

    class gfc3d:
        def __init__(self, filename):
            with h5py.File(filename, 'r+') as hdf5_file:
                if 'fclib_global' in hdf5_file:
                    m = hdf5_file['fclib_global']['M']['m'][0]
                    n = hdf5_file['fclib_global']['M']['n'][0]
                    nz = hdf5_file['fclib_global']['M']['nz'][0]
                    nzmax = hdf5_file['fclib_global']['M']['nzmax'][0]
                    i = hdf5_file['fclib_global']['M']['i']
                    p = hdf5_file['fclib_global']['M']['p']
                    x = hdf5_file['fclib_global']['M']['x']
                    if (nz > 0):
                        from scipy.sparse import coo_matrix
                        self.M = coo_matrix((x, (i, p)), shape=(m, n))
                    elif (nz == -1):
                        from scipy.sparse import csc_matrix
                        self.M = csc_matrix((x, (i, p)), shape=(m, n))
                    elif (nz == -2):
                        from scipy.sparse import csr_matrix
                        self.M = csr_matrix((x, (i, p)), shape=(m, n))
                    else:
                        print('unknown matrix format')

                    m = hdf5_file['fclib_global']['H']['m'][0]
                    n = hdf5_file['fclib_global']['H']['n'][0]
                    nz = hdf5_file['fclib_global']['H']['nz'][0]
                    nzmax = hdf5_file['fclib_global']['H']['nzmax'][0]
                    i = hdf5_file['fclib_global']['H']['i']
                    p = hdf5_file['fclib_global']['H']['p']
                    x = hdf5_file['fclib_global']['H']['x']
                    if (nz > 0):
                        from scipy.sparse import coo_matrix
                        self.H = coo_matrix((x, (i, p)), shape=(m, n))
                    elif (nz == -1):
                        from scipy.sparse import csc_matrix
                        self.H = csc_matrix((x, (i, p)), shape=(m, n))
                    elif (nz == -2):
                        from scipy.sparse import csr_matrix
                        self.H = csr_matrix((x, (i, p)), shape=(m, n))
                    else:
                        print('unknown matrix format')

                    self.f = numpy.array(hdf5_file['fclib_global']['vectors']['f'])
                    self.w = numpy.array(hdf5_file['fclib_global']['vectors']['w'])
                    self.mu = numpy.array(hdf5_file['fclib_global']['vectors']['mu'])

                else:
                    print('No global problem in', filename)
                    pass

        def __str__(self):
            str_1 = 'matrix M :' + str(self.M) + '\n' \
                    + 'matrix H :' + str(self.H) + '\n' \
                    + 'vector f : ' + str(self.f) + '\n' \
                    + 'vector w : ' + str(self.w) + '\n' \
                    + 'vector mu : ' + str(self.mu) + '\n'
            return str_1

    problem = gfc3d(filename)

    return problem

    # dense version of the matrices
    # Mdense = problem.M.todense()
    # Hdense = problem.H.todense()
