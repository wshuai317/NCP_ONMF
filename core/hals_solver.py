###############################################################################
# The script is to use a solver based on HALS method to solve the ONMF problem
#     min_{F, G} || X - FG^{T}||_{F}^{2}
#      s.t.  F >= 0, G >= 0, F^{T}F = I
#
# @author Wang Shuai
# @date 2018.10.15
###############################################################################

from __future__ import division
from convergence import Convergence
import numpy as np
from numpy import linalg as LA
#from sklearn.cluster import KMeans
#from sklearn.decomposition import NMF
import os.path as path
#import random
import time

class HALS_Solver(object):
    def __init__(self, data_manager = None, res_dir = None, rank = 4, seed_num = 1):
        if data_manager is None or res_dir is None:
            raise ValueError('Error: some inputs are missing!')
	self.data_manager = data_manager
	self.data_mat = self.data_manager.get_data_mat()
	self.data_mat = np.asmatrix(np.copy(self.data_mat).transpose())
        W_init, H_init = self.data_manager.gen_inits_WH(init = 'random', seed = seed_num, H_ortho = False)
        self.F, self.G = H_init.transpose(), W_init
        self.res_dir = res_dir
        self.rank = rank
        #self.SNR = SNR
        self.seed_num = seed_num
       	self.converge = Convergence(res_dir)
        #np.random.seed(seed_num)  # set the seed so that each run will get the same initial values
        (m, n) = self.data_mat.shape
	self.flag = 0   # flag to indicate whether G can be negative or not
			# flag = 0 : the G should be nonnegative 
			# flag = 1: the G can be negative
        #self.n_factor = m * n # set the normalization factor to normalize the objective value
        self.n_factor = LA.norm(self.data_mat, 'fro') ** 2
        self.time_used = 0 # record the time used by the method
        self.U = None  # used  for update F

    def set_max_iters(self, num):
        self.converge.set_max_iters(num)

    def set_tol(self, tol):
        self.converge.set_tolerance(tol)

    def get_obj_val(self):
        res = LA.norm(self.data_mat - self.F * self.G.transpose(), 'fro')**2 / self.n_factor
        return res

    def get_iter_num(self):
        return self.converge.len()

    def update_F(self):
        A = self.data_mat * self.G
        B = self.G.transpose() * self.G
        #print self.F.shape
        #print self.G.shape
        for j in range(self.rank):
            Fj = self.U - self.F[:, j]
            #print 'B[ji]'
            #print B[j, j]
            h = A[:, j] - self.F * B[:, j] + B[j, j] * self.F[:, j]
            #print 'Fj' + str(Fj.shape)
            #print 'h' + str(h.shape)
            #print 'Fj * Fj' + str(Fj.transpose() * Fj)
	    tmp = np.multiply(Fj.transpose() * h, Fj) / np.asscalar(Fj.transpose() * Fj)
	    tmp = h - tmp
            fj = np.maximum(1e-30, tmp)
            #print (Fj.transpose() * Fj)[0, 0]
            #print fj
            #print LA.norm(fj, 2)
            fj = fj / LA.norm(fj, 2)
            self.F[:, j] = fj
            self.U = Fj + fj

    def update_G(self):
        C = self.data_mat.transpose() * self.F
        D = self.F.transpose() * self.F
        #print D
        for j in range(self.rank):
            if self.flag == 0:
		temp = C[:, j] - self.G * D[:, j] + D[j, j] * self.G[:, j]
		self.G[:, j] = np.maximum(temp, 1e-30)
	    else:
		self.G[:, j] = C[:, j] - self.G * D[:, j] + D[j, j] * self.G[:, j]

    def solve(self):

        obj_val = self.get_obj_val()
        print 'The initial error: iter = ' + str(self.get_iter_num()) + ', obj =' + str(obj_val)
        self.converge.add_obj_value(obj_val)
        self.converge.add_prim_value('F', self.F)
        self.converge.add_prim_value('G', self.G)

        # initialize U
	start_time = time.time()
        self.U = self.F * np.asmatrix(np.ones(self.rank)).transpose()
	end_time = time.time()
	self.time_used += end_time - start_time
        #print self.F[0, :]
        #print self.U[0]

        print 'Start to solve the problem by HALS ONMF ----------'
        while not self.converge.d():
            # update the variable W , H iteratively according to DTPP method
            start_time = time.time()  # record the start time
            self.update_F()
            self.update_G()
            end_time = time.time()   # record the end time
            self.time_used += end_time - start_time
            #print self.F[0:5, 0:5]
            # store the newly obtained values for convergence analysis
            self.converge.add_prim_value('F', self.F)
            self.converge.add_prim_value('G', self.G)

            # store the objective function value
            obj_val = self.get_obj_val()
            self.converge.add_obj_value(obj_val)
            print 'onmf_HALS: iter = ' + str(self.get_iter_num()) + ', obj = ' + str(obj_val)

            # store the satisfaction of feasible conditions
            #(ha, hb) = self.F.shape
            fea = LA.norm(self.F.transpose() * self.F - np.asmatrix(np.eye(self.rank)), 'fro') / (self.rank * self.rank)
            self.converge.add_fea_condition_value('FTF_I', fea)


        print 'Stop the solve the problem ---------'
        self.converge_analysis()


    ''' return the solution W, H '''
    def get_solution(self):
        return self.G, self.F.transpose()

    ''' return the optimal obj val '''
    def get_opt_obj_and_fea(self):
        return self.get_obj_val(), self.converge.get_last_fea_condition_value('FTF_I')

    ''' return the iteration number and time used '''
    def get_iter_and_time(self):
        return self.get_iter_num(), self.time_used

    def get_time(self):
	return self.time_used
	
    ''' return the cluster assignment from H '''
    def get_cls_assignment_from_H(self):
        labels = np.argmax(np.asarray(self.F), 1)
	#print len(labels)
	#print self.data_mat.shape[1]
        if len(labels) != self.data_mat.shape[0]:
            raise ValueError('Error: the size of data samples must = the length of labels!')
        return labels

    ''' simulation result analysis (convergence plot) '''
    def converge_analysis(self):
        # get the dirname to store the result: data file and figure
	m_name = 'hals_' + str(self.flag)
        dir_name = path.join(self.res_dir, 'onmf', m_name, 'rank' + str(self.rank), self.data_manager.get_data_name(), 'seed' + str(self.seed_num))
        print 'Start to plot and store the obj convergence ------'
        self.converge.plot_convergence_obj(dir_name)
        print 'Start to plot and store the primal change ------'
        self.converge.plot_convergence_prim_var(dir_name)
        print 'Start to plot and store the fea condition change ------'
        self.converge.plot_convergence_fea_condition(dir_name)
        print 'Start to store the obj values and the factors'
        self.converge.store_obj_val(dir_name)
        self.converge.store_prim_val(-1, dir_name) # store the last element of primal variable
