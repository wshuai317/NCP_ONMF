# This script is the core of the project defining the NMF class to perform the
# nonnegative matrix factorization and return the factors
#
#           min  D(X|WH) s.t. W >= 0, H >= 0
#
# Note:
#    A.there are many combinations of distance function D() and optimization methods for nmf.
#       Now we only focus on the Euclidean distance and the following two methods:
#           0. multiplicative rules
#           1. palm (proximal alternating linearized minimization)
#    B. the input data matrix should not contain negative values and all 0s rows
#
# Author: Wang Shuai
# Time: 2018.02.23
###############################################################################
from __future__ import division  # change the result of division to float type
#from convergence import Convergence
import numpy as np
from numpy import linalg as LA
from sklearn.cluster import KMeans as sklearn_KMeans
from sklearn.decomposition import NMF as sklearn_NMF
#import random
#import pandas as pd
import matplotlib.pyplot as plt
from plam_solver import PLAM_Solver
from nmf_solver import NMF_Solver

class NMF(object):
    def __init__(self, data_manager = None, res_dir = None, rank = 4, seed_num = 1):
        if data_manager is None or res_dir is None:
            raise ValueError('Error: some inputs are missing!')
        self.data_manager = data_manager
        self.res_dir = res_dir
        self.rank = rank
        self.SNR = 1
        self.seed_num = seed_num
        #np.random.seed(seed_num)  # set the seed so that each run will get the same initial values
        #random.seed(seed_num)
        #print 'seed_num for NMF: ' + str(seed_num)
        #self.n_factor = 1 # default value of normalization factor = 1
        self.init = 'random' # the way to initialize W, H

    def factors(self):
        ''' return W, H as matries'''
        return self.W, self.H

    def solve(self, m_name = 'mul_rule'):
        '''
        perform the nonnegative matrix factorization
        We have two methods (solvers):
            0: multiplicative rules
            1: palm (proximal alternating linearized minimization)
        '''
        #(W_init, H_init) = self.initialize_WH(self.init)
	#(W_init, H_init) = self.data_manager.gen_inits_WH(init = self.init, seed = self.seed_num, H_ortho = False)
        if m_name == 'mul_rule':
            solver = NMF_Solver(self.data_manager, self.res_dir, self.rank, self.seed_num)
            solver.set_max_iters(2000)
            solver.set_tol(1e-5)
            solver.solve()
            (self.W, self.H) = solver.get_solution()
        elif m_name == 'palm':
            solver = PLAM_Solver(self.data_manager, self.res_dir, self.rank, self.seed_num)
            solver.set_max_iters(2000)
            solver.set_tol(1e-5)
            solver.solve()
            (self.W, self.H) = solver.get_solution()
        else:
            print 'Error: no other methods to be used for nmf!'
        return solver.get_cls_assignment_from_H(), solver.get_time()

if __name__ == "__main__":
    # test the generatation of synthetic data
    '''
    dim = 2
    rank = 4
    num_list = [100, 100, 100, 100]
    mean_list = [[0, 0], [4, 4], [1, 4], [4, 1]]
    #cov_list = []
    #cov_list.append([[1, 0], [0, 1]])
    #cov_list.append([[1, 0], [0, 1]])
    cov_list = []
    for i in range(len(num_list)):
        #cov_list.append(np.eye(dim) * np.random.uniform(0, 0.5))
        cov_list.append(np.eye(dim) * 0.4)


    nmf = NMF('~/Work/research/python_code/nmf/', '~/Work/research/python_code/nmf/', rank)
    data_mat = nmf.gen_synthetic_data(dim, num_list, mean_list, cov_list)
    print data_mat.shape
    x = data_mat[0, :]
    #print x
    y = data_mat[1, :]
    '''
    np.random.seed(1)
    dim = 2
    data_num = 1000
    rank = 3
    num_list = np.random.randint(0.2 * data_num / rank, 1.2 * data_num / rank, rank - 1).tolist()
    num_list.append(data_num - np.sum(num_list))
    print 'num_list'
    print num_list
    mean_list = np.random.uniform(0, 100, size = (rank, dim)).tolist()
    cov_list = []
    for i in range(rank):
        cov_list.append(np.eye(dim) * 20)

    nmf = NMF('~/Work/research/python_code/nmf/', '~/Work/research/python_code/nmf/', rank)
    data_mat = nmf.gen_synthetic_data(dim, num_list, mean_list, cov_list)
    print data_mat.shape
    x = data_mat[0, :]
    y = data_mat[1, :]
    #print y
    #print data_mat
    plt.plot(x, y, marker = '+', markersize = 5, color = 'b', linewidth = 3)
    #plt.axis('equal')
    plt.show()


