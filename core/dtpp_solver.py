###############################################################################
# The script is to use a solver based on DTPP method to solve the ONMF problem
#
# @author Wang Shuai
# @date 2018.05.08
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

class DTPP_Solver(object):
    def __init__(self, data_manager = None, res_dir = None, rank = 4, seed_num = 1):
        if data_manager is None or res_dir is None:
            raise ValueError('Error: some inputs are missing!')
        self.data_manager = data_manager
        self.W, self.H = self.data_manager.gen_inits_WH(init = 'random', seed = seed_num, H_ortho = False)
	self.data_mat = self.data_manager.get_data_mat()
	#self.true_labels = self.data_manager.get_labels()  
        self.res_dir = res_dir
        self.rank = rank
        self.seed_num = seed_num
        self.converge = Convergence(res_dir)
        np.random.seed(seed_num)  # set the seed so that each run will get the same initial values
        (m, n) = self.data_mat.shape
        #self.n_factor = m * n # set the normalization factor to normalize the objective value
        self.n_factor = LA.norm(self.data_mat, 'fro') ** 2
        self.time_used = 0 # record the time used by the method
	#self.set_max_iters(1000)

    def set_max_iters(self, num):
        self.converge.set_max_iters(num)

    def set_tol(self, tol):
        self.converge.set_tolerance(tol)

    def get_obj_val(self):
	#print self.data_mat.shape
	#print self.W.shape
	#print self.H.shape
        res = LA.norm(self.data_mat - self.W * self.H, 'fro')**2 / self.n_factor
        return res

    def get_iter_num(self):
        return self.converge.len()

    def update_prim_var(self, var_name):
        if var_name == 'W':
            #W = W .* ((V * H') ./ max(W * (H * H'), myeps));
            temp = np.divide(self.data_mat * self.H.transpose(), \
                    np.maximum(self.W * (self.H * self.H.transpose()), 1e-20))
            self.W = np.multiply(self.W, temp)
        elif var_name == 'H':
            #H = H .* (((W' * V) ./ max(W' * V * (H' * H), myeps)) .^ (1/2));
            temp = np.divide(self.W.transpose() * self.data_mat, np.maximum(self.W.transpose() * self.data_mat * (self.H.transpose() * self.H), \
                    1e-20))
            self.H = np.multiply(self.H, np.power(temp, 0.5))
        else:
            raise ValueError('Error: no other variable should be updated!')

    def solve(self):

        obj_val = self.get_obj_val()
        print 'The initial error: iter = ' + str(self.get_iter_num()) + ', obj =' + str(obj_val)
	#print 'max_iter: ' + str(self.
        self.converge.add_obj_value(obj_val)
        self.converge.add_prim_value('W', self.W)
        self.converge.add_prim_value('H', self.H)

        print 'H0'
        print self.H

        print 'Start to solve the problem by DTPP ----------'
        while not self.converge.d():
            # update the variable W , H iteratively according to DTPP method
            start_time = time.time()  # record the start time
            self.update_prim_var('W')
            self.update_prim_var('H')
            end_time = time.time()   # record the end time
            self.time_used += end_time - start_time

            # store the newly obtained values for convergence analysis
            self.converge.add_prim_value('W', self.W)
            self.converge.add_prim_value('H', self.H)

            # store the objective function value
            obj_val = self.get_obj_val()
            self.converge.add_obj_value(obj_val)
            print 'onmf_DTPP: iter = ' + str(self.get_iter_num()) + ', obj = ' + str(obj_val)

            # store the satisfaction of feasible conditions
            (ha, hb) = self.H.shape
            fea = LA.norm(self.H * self.H.transpose() - np.asmatrix(np.eye(ha)), 'fro') / (ha * ha)
            self.converge.add_fea_condition_value('HTH_I', fea)


        print 'Stop the solve the problem ---------'
        self.converge_analysis()


    ''' return the solution W, H '''
    def get_solution(self):
        return self.W, self.H

    ''' return the optimal obj val '''
    def get_opt_obj_and_fea(self):
        return self.get_obj_val(), self.converge.get_last_fea_condition_value('HTH_I')

    ''' return the iteration number and time used '''
    def get_iter_and_time(self):
        return self.get_iter_num(), self.time_used

    def get_time(self):
	return self.time_used

    ''' return the cluster assignment from H '''
    def get_cls_assignment_from_H(self):
        labels = np.argmax(np.asarray(self.H), 0)
        if len(labels) != self.data_mat.shape[1]:
            raise ValueError('Error: the size of data samples must = the length of labels!')
        return labels
	
    ''' simulation result analysis (convergence plot) '''
    def converge_analysis(self):
        # get the dirname to store the result: data file and figure
        dir_name = path.join(self.res_dir, 'onmf', 'dtpp', 'rank' + str(self.rank), self.data_manager.get_data_name(), 'seed' + str(self.seed_num))
        print 'Start to plot and store the obj convergence ------'
        self.converge.plot_convergence_obj(dir_name)
        print 'Start to plot and store the primal change ------'
        self.converge.plot_convergence_prim_var(dir_name)
        print 'Start to plot and store the fea condition change ------'
        self.converge.plot_convergence_fea_condition(dir_name)
        print 'Start to store the obj values and the factors'
        self.converge.store_obj_val(dir_name)
        self.converge.store_prim_val(-1, dir_name) # store the last element of primal variabl

