##############################################################################
# The script is to solve the onmf problems using ONP_MF method.
#       min ||X - W * H||_{F}^{2}
#       s.t. W >= 0, H >= 0
#            H * H' = I.
#
# @author Wang Shuai
# @date 2018.05.01
##############################################################################
from __future__ import division
from convergence import Convergence
import numpy as np
from numpy import linalg as LA
#from sklearn.cluster import KMeans
#from sklearn.decomposition import NMF
import os.path as path
#import random
import time
import scipy.optimize as optm
import scipy

class ONPMF_Solver(object):

    def __init__(self, data_manager = None, res_dir = None, rank = 4, seed_num = 1):
        if data_manager is None or res_dir is None:
            raise ValueError('Error: input is missing!')
        self.rank = rank
        self.res_dir = res_dir
        self.seed_num = seed_num
        self.converge = Convergence(res_dir)
        self.data_manager = data_manager
	self.data_mat = self.data_manager.get_data_mat()
        self.W, self.H = self.data_manager.gen_inits_WH(init = 'random', seed = seed_num, H_ortho = True)
	self.W = np.asmatrix(self.W, dtype = np.float64)
	self.H = np.asmatrix(self.H, dtype = np.float64)
        #np.random.seed(seed_num)  # set the seed so that each run will get the same initial values
        #(m, n) = self.data_mat.shape
        self.n_factor = LA.norm(self.data_mat, 'fro') ** 2 # set the normalization factor to normalize the objective value
        self.time_used = 0
	self.flag = 0 # the flag indicates whether the W can be negative or not depending on the data
		      # flag = 0 : W must be nonnegative
		      # flag = 1: W can be negative


    def set_max_iters(self, num):
        self.converge.set_max_iters(num)

    def set_tol(self, tol):
        self.converge.set_tolerance(tol)


    ''' initialize the primal variables '''
    def initialize_prim_vars(self):
        self.W = np.asmatrix(self.W)
        '''
        U, s, V = LA.svd(self.data_mat)
        self.H = np.asmatrix(V[0:10, :])
        
        print self.H * self.H.transpose()
         
	(ha, hb) = self.H.shape
        labels = np.argmax(np.asarray(np.abs(self.H)), 0)
 	H = np.zeros((ha, hb))
        for j in range(hb):
            H[labels[j], j] = 1
        H = np.asmatrix(H)
        H = np.asmatrix(np.diag(np.diag(H * H.transpose()) ** (-0.5))) * H
        print H * H.transpose()
	self.H = H
        '''
	# self.H = np.maximum(self.H, 0)
	
        self.converge.add_prim_value('W', self.W) # store the initial values
        self.converge.add_prim_value('H', self.H)


    def initialize_dual_vars(self):
        (m, n) = self.data_mat.shape
        self.Z = np.asmatrix(np.zeros(shape = (self.rank, n), dtype = np.float64))
        self.converge.add_dual_value('Z', self.Z) # store the initial value for dual variables

    def initialize_penalty_para(self, flag = 0):
        # set the penalty parameter rol
        self.rol = 0.01
        self.alpha = 100
        self.gamma = 1.01
	'''
	if self.data_manager.get_data_name() == 'mnist#8':
	    self.scale = 0.001
	else: 
	    self.scale = 0.00001
	'''
	self.scale = 0.00001
	if self.flag == 0:
	    self.mul = 0
	else: self.mul = 1e-10  # used when U can be negative 

    ''' compute the lagrangian function value for testing '''
    def get_lag_val(self):
        sum_t = LA.norm(self.data_mat - self.W * self.H, 'fro') ** 2 / 2 \
                + self.mul * LA.norm(self.W, 'fro')**2 + np.trace(self.Z.transpose() * (-self.H)) + \
                0.5 * self.rol * LA.norm(np.minimum(self.H, 0), 'fro')**2
        return sum_t


    def get_obj_val(self):
        res = LA.norm(self.data_mat - self.W * self.H, 'fro')**2 / self.n_factor
        return res

    def get_iter_num(self):
        return self.converge.len()

    ''' update the primal variable with a given name at each iteration of ADMM'''
    def update_prim_var(self, var_name):
        if var_name == 'H': # update primal variable Y
            beta = 0.01
            step_size = 1
            gradient = self.W.transpose() * (self.W * self.H - self.data_mat) - self.Z + self.rol * np.minimum(0, self.H)
	    #print (gradient)
	    #print (self.W)
	    #print (self.Z)
	    #print (self.rol)
            Lx = 0.5 * LA.norm(self.data_mat - self.W * self.H, 'fro')**2 
            Lx = Lx + np.trace(self.Z.transpose() * (-self.H)) 
	    Lx = Lx + 0.5 * self.rol * LA.norm(np.minimum(self.H, 0), 'fro') ** 2
            #print 'Lx: ' + str(Lx)
            while True:
                B = self.H - step_size * gradient
                #U, s, V = LA.svd(B)
		#print (B)
		#print (self.H)
		#print (step_size)
		#print (gradient)
		U, s, V = scipy.linalg.svd(B)
                (a, b) = B.shape
                E = np.asmatrix(np.eye(a, b))
                H_new = np.asmatrix(U) * E * np.asmatrix(V)  # pay attention V do not to be transposed
                #H_new = H_new.transpose()   # this is very important
                Lz = 0.5 * LA.norm(self.data_mat - self.W * H_new, 'fro')**2 \
                        + np.trace(self.Z.transpose() * (-H_new)) \
                        + 0.5 * self.rol * LA.norm(np.minimum(H_new, 0), 'fro') ** 2
                #print 'update H : ' + str(step_size) + ' ' + str(Lz)
                if Lz <= Lx + self.scale * np.trace(gradient.transpose() * (H_new - self.H)):
                    break
                step_size = step_size * beta
            self.H = np.asmatrix(np.copy(H_new))

        elif var_name == 'W': # update primal variable P
	    if self.flag == 0:
		'''
                for j in range(10):
                    beta = 0.1
                    step_size = 1
                    gradient = (self.W * self.H - self.data_mat) * self.H.transpose()
                    f_x = LA.norm(self.data_mat - self.W * self.H, 'fro')**2 / 2
                    while True:
                        W_new = np.maximum(0, self.W - step_size * gradient)
                        f_z = LA.norm(self.data_mat - W_new * self.H, 'fro') ** 2 / 2
                        if f_z <= f_x + 0.00001 * np.trace(gradient.transpose() * (W_new - self.W)):
                            break
                        step_size = step_size * beta
                    test = LA.norm(W_new - self.W, 'fro') / LA.norm(self.W, 'fro')
                    self.W = np.asmatrix(np.copy(W_new))
                    if test < 1e-5:
                        #print 'satisfy the stopping criteria!'
                        break	
		'''
		(wa, wb) = self.W.shape
		for i in range(wa):
		    b = np.array(self.data_mat[i, :]).flatten()
		    t, bla = optm.nnls(self.H.transpose(), b)
                    self.W[i, :] = np.asmatrix(t)
	    else:
		(ha, hb) = self.H.shape
                I_ha = np.asmatrix(np.eye(ha))
		self.W = self.data_mat * self.H.transpose() * LA.inv(self.H * self.H.transpose() + self.mul * I_ha)
	        	
        else:
            raise ValueError('Error: no primal variable with this name to be updated!')

    ''' update the dual variable with a given name at each iteration of ADMM'''
    def update_dual_var(self, var_name):
        if var_name == 'Z': # update dual variable Z
            k = np.maximum(self.get_iter_num(), 1)
            self.Z = np.maximum(0, self.Z - (self.alpha / k) * self.H)
        else:
            raise ValueError('Error: no dual variable with this name to be updated!')

    ''' update the penalty parameters rol1 and rol2 adaptively '''
    def update_penalty_parameters(self):
        self.rol = self.gamma * self.rol

    def solve(self):
        self.initialize_prim_vars()
        self.initialize_dual_vars()
        self.initialize_penalty_para()
	data_name = self.data_manager.get_data_name()
	if data_name.startswith('tdt2') or data_name.startswith('tcga'):
	    self.set_max_iters(500)
	#self.set_max_iters(500)
	print self.H[:, 0]

        #obj_val = self.get_obj_val()
        #print 'The initial error: iter' + str(self.get_iter_num()) + ', obj =' + str(obj_val)
        print 'Start to solve the problem by ADMM ------------'
        while not self.converge.d():
            start_time = time.time()

            # update the primal and dual variables accroding to ADMM algorithm
            #print 'befor update, lag_val: ' + str(self.get_lag_val())
            # update primal variable Y
            self.update_prim_var('W')
            #print 'after update W, lag_val: ' + str(self.get_lag_val())
            # update primal variable H
            self.update_prim_var('H')
            #print 'after update H, lag_val: ' + str(self.get_lag_val())

            # update dual varialbe Z
            self.update_dual_var('Z')

            self.update_penalty_parameters()

            end_time = time.time()
            self.time_used += end_time - start_time


            # store the newly obtained values for convergence analysis
            # note that the change of each primal and dual varialbes will also be computed and added if any
            self.converge.add_prim_value('W', self.W)
            self.converge.add_prim_value('H', self.H)
            self.converge.add_dual_value('Z', self.Z)
		
            
            obj_val = self.get_obj_val()
            self.converge.add_obj_value(obj_val)
            print 'onmf_onpmf: iter = ' + str(self.get_iter_num()) + ', obj = ' + str(obj_val)
	    
	    #print 'onmf_onpmf: iter = ' + str(self.get_iter_num()) 
            # store the satisfaction of feasible conditions
            (ha, hb) = self.H.shape
            fea = LA.norm(self.H * self.H.transpose() - np.asmatrix(np.eye(ha)), 'fro') / (ha * ha)
            self.converge.add_fea_condition_value('HTH_I', fea)

        print 'Stop to solve the problem ------'
        self.converge_analysis()

    ''' return the solution W, H '''
    def get_solution(self):
        return self.W, self.H

    ''' return the optimal objective value and feasibility level '''
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
        #dir_name = path.join(self.res_dir, path.basename(path.normpath(self.data_path)), 'rank' + str(self.rank))
	m_name = 'onp_mf1_' + str(self.flag)
        dir_name = path.join(self.res_dir, 'onmf', m_name, 'alpha100', 'rank' + str(self.rank), self.data_manager.get_data_name(), 'seed' + str(self.seed_num))
        print 'Start to plot and store the obj convergence ------'
        self.converge.plot_convergence_obj(dir_name)
        print 'Start to plot and store the primal change ------'
        self.converge.plot_convergence_prim_var(dir_name)
        print 'Start to plot and store the dual change ------'
        self.converge.plot_convergence_dual_var(dir_name)
        print 'Start to plot and store the fea condition change ------'
        self.converge.plot_convergence_fea_condition(dir_name)
        print 'Start to store the obj values and the factors'
        self.converge.store_obj_val(dir_name)
        self.converge.store_prim_val(-1, dir_name) # store the last element of primal varialbes list
