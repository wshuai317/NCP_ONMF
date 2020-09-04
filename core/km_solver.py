###############################################################################
# The script is to use a solver based on update rules of K-Means to solve the ONMF problem
#          min ||X- WH||_{F}^2
#          s.t. |h_j|_0 = 1
#               W >= 0, H >= 0.
#
# @author Wang Shuai
# @date 2019.02.21
###############################################################################

from __future__ import division
from convergence import Convergence
import numpy as np
from numpy import linalg as LA
import os.path as path
import time
from utils import *

class KM_Solver(object):
    def __init__(self, data_mat = None, W = None, H = None, res_dir = None, rank = 4, SNR = -5, seed_num = 1, true_labels = None):
        if data_mat is None or W is None or H is None or res_dir is None:
            raise ValueError('Error: some inputs are missing!')
        self.data_mat = data_mat
        self.W, self.H = W, H
        self.rank = rank
        self.SNR = SNR
        self.res_dir = res_dir
        self.seed_num = seed_num
        self.converge = Convergence(res_dir)
	self.labels = true_labels
        np.random.seed(seed_num)  # set the seed so that each run will get the same initial values
        (m, n) = self.data_mat.shape
	self.flag = 0 # flag to indicate whether to use LS or gradient descent to update W
	m_name = 'km' + str(self.flag)
        self.output_dir = path.join(self.res_dir, 'onmf', m_name, 'rank' + str(self.rank), 'data' + str(SNR), 'seed' + str(self.seed_num))
        self.time_used = 0 # record the time elapsed when running the simulations
       

    def set_max_iters(self, num):
        self.converge.set_max_iters(num)

    def set_tol(self, tol):
        self.converge.set_tolerance(tol)

    def get_nmf_cost(self, W, H):
        res = LA.norm(self.data_mat - W * H, 'fro')**2
        return res

    def get_iter_num(self):
        return self.converge.len()
    

    def update_scheme(self, verbose = False):
        '''
        The function performs the update of W and H using similar ways as that in K-means
	Specifically,
		for each column of H, h_j,
			it will try to consider each pos as the place holding non-zero entry
			For example, if H_{k, j} != 0, then 
				H_{k, j} = arg min_{c > 0} ||x_j - w_k * c||_2^2
			which leads to
				H_{k, j} = (x_j^T w_k) / ||w_k||_2^2 if w_k != 0, and otherwise, = 1
                        By trying all k = 1, ..., K. we obtain h_j with lowest obj value
		for each column of W, w_k,
			w_k = arg min_{ w >= 0} sum_{j \in C_k} (x_j - w * H_{k, j})^2
			which leads to 
				w_k = X * ~h_k^T / ||~h_k||_2^2
				
        '''
        # update H
	#p_cost = self.get_nmf_cost(self.W, self.H)
	(ha, hb) = self.H.shape
	H_pre = np.asmatrix(np.copy(self.H))
	for j in range(hb):
	    tmp = LA.norm(self.data_mat[:, j] - self.W * H_pre[:, j], 2) ** 2
	    p_cost = self.get_nmf_cost(self.W, self.H)
	    for k in range(ha):
	        h_j_new = np.asmatrix(np.zeros((ha, 1)))
		#print h_j_new
	        #print h_j_new
		if LA.norm(self.W[:, k], 2) == 0: 
	            #print 'the k th column of W is 0'
		    h_j_new[k, 0] = 1
		else: h_j_new[k, 0] = self.data_mat[:, j].transpose() * self.W[:, k] / (LA.norm(self.W[:, k], 2) ** 2)
		# check if a smaller obj value is obtained
		val = LA.norm(self.data_mat[:, j] - self.W * h_j_new, 2) ** 2
		#print 'val: ' + str(val) + ', tmp: ' +str(tmp)
		if val < tmp:
		   self.H[:, j] = np.copy(h_j_new)
		   tmp = val
	    '''
	    c_cost = self.get_nmf_cost(self.W, self.H)
	    if c_cost > p_cost:
		print 'cur cost: ' + str(c_cost) + ', p_cost: ' + str(p_cost)
		print H_pre[:, j]
		print self.H[:, j]
		print LA.norm(self.data_mat[:, j] - self.W * H_pre[:, j], 'fro') ** 2
		print LA.norm(self.data_mat[:, j] - self.W * self.H[:, j], 'fro') ** 2
		print '------'
		print LA.norm(self.data_mat[:, 0:2] - self.W * H_pre[:, 0:2], 'fro') ** 2
		print LA.norm(self.data_mat[:, 0:2] - self.W * self.H[:, 0:2], 'fro') ** 2
		
		print '------'
                print LA.norm(self.data_mat[:, 0] - self.W * H_pre[:, 0], 'fro') ** 2
                print LA.norm(self.data_mat[:, 0] - self.W * self.H[:, 0], 'fro') ** 2        
		
		print '------'
                print LA.norm(self.data_mat[:, 1] - self.W * H_pre[:, 1], 'fro') ** 2
                print LA.norm(self.data_mat[:, 1] - self.W * self.H[:, 1], 'fro') ** 2
		raise ValueError('Error: j = ' + str(j))
		#print self.H[:, j]
	     '''
	if verbose:
	   print 'KM: iter = ' + str(self.get_iter_num()) + ', after update H -' + \
                        ', nmf cost = ' + str(self.get_nmf_cost(self.W, self.H))
	   #c_cost = self.get_nmf_cost(self.W, self.H)
	   #if c_cost > p_cost:
	   #    print self.H
	   #    raise ValueError('Error')
	# update W
	if self.flag == 0:  # use the LS or K-means way to update W (centroids) 
	    for k in range(ha):
	        if LA.norm(self.H[k, :], 2) == 0: # if no data points belongs to cluster k
	            self.W[:, k].fill(0)
		else:
	            self.W[:, k] = self.data_mat * self.H[k, :].transpose() / (LA.norm(self.H[k, :], 2) ** 2)
	else: # use the gradient descent to update W
	    Hessian = self.H * self.H.transpose()
            #c = 0.5 * LA.norm(Hessian, 'fro')
            egenvals, _ = LA.eigh(Hessian)
            c = 0.51 * np.max(egenvals)
            grad_W_pre = self.W * Hessian - self.data_mat * self.H.transpose()
            self.W = np.maximum(0, self.W - grad_W_pre / c)
	
	if verbose:
	   print 'KM: iter = ' + str(self.get_iter_num()) + ', after update W -' + \
			', nmf cost = ' + str(self.get_nmf_cost(self.W, self.H))		
        

    def solve(self):
        start_time = time.time()
        self.set_tol(1e-5)
	end_time = time.time()
	self.time_used += end_time - start_time
        
        #cost = self.get_nmf_cost(self.W, self.H)
	cost = self.get_nmf_cost(self.W, self.H)
        #self.converge.add_obj_value(cost)
        self.converge.add_obj_value(cost)
        self.converge.add_prim_value('W', self.W)
        self.converge.add_prim_value('H', self.H)

        print self.H[:, 0]

        acc_km = []  # record the clustering accuracy for each SNCP iteration
        time_km = [] # record the time used after each SNCP iteration
       

	# calculate the clustering accurary
        pre_labels = np.argmax(np.asarray(self.H), 0)
        if self.labels is None:
	    raise ValueError('Error: no labels!')
        acc = calculate_accuracy(pre_labels, self.labels)
        acc_km.append(acc)


	time_km.append(self.time_used)

        print 'Start to solve the problem by KM ----------'
        while not self.converge.d():

            # update the variable W , H
	    start_time = time.time()
            self.update_scheme(verbose = False)
	    end_time = time.time()
	    self.time_used += end_time - start_time
	    time_km.append(self.time_used)
	    print 'time used: ' + str(self.time_used)

	    # calculate the clustering accurary
	    pre_labels = np.argmax(np.asarray(self.H), 0)
	    if self.labels is None:
		raise ValueError('Error: no labels!')
	    acc = calculate_accuracy(pre_labels, self.labels)
	    acc_km.append(acc)
	    

            # store the newly obtained values for convergence analysis
            self.converge.add_prim_value('W', self.W)
            self.converge.add_prim_value('H', self.H)
		            
            # store the obj_val
	    cost = self.get_nmf_cost(self.W, self.H)
            self.converge.add_obj_value(cost)
	   
            print 'onmf_KM: iter = ' + str(self.get_iter_num()) + ', nmf_cost = ' + str(cost)

        print 'HTH:'
        print self.H * self.H.transpose()


        # show the number of inner iterations
        self.converge.save_data(time_km, self.output_dir, 'time_km.csv')
        self.converge.save_data(acc_km, self.output_dir, 'acc_km.csv')
        

        print 'Stop the solve the problem ---------'
        self.converge_analysis()

    ''' return the solution W, H '''
    def get_solution(self):
        return self.W, self.H

    ''' return the optimal obj val '''
    def get_opt_obj_and_fea(self):
        return self.get_nmf_cost(self.W, self.H), None

    ''' return the iteration number and time used '''
    def get_iter_and_time(self):
        return self.get_iter_num(), self.time_used

    ''' simulation result analysis (convergence plot) '''
    def converge_analysis(self):
        # get the dirname to store the result: data file and figure
        #dir_name = path.join(self.res_dir, 'onmf', 'penalty', 'inner<1e-3', 'rank' + str(self.rank), 'SNR-3', 'seed' + str(self.seed_num))
        dir_name = self.output_dir
        print 'Start to plot and store the obj convergence ------'
        self.converge.plot_convergence_obj(dir_name)
        print 'Start to plot and store the primal change ------'
        self.converge.plot_convergence_prim_var(dir_name)
        print 'Start to plot and store the fea condition change ------'
        self.converge.plot_convergence_fea_condition(dir_name)
        print 'Start to store the obj values and the factors'
        self.converge.store_obj_val(dir_name)
        self.converge.store_prim_val(-1, dir_name) # store the last element of primal variabl

