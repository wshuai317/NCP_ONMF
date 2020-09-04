###############################################################################
# The script is to use a solver based on SNCP method to solve the ONMF problem
#          min 0.5 * ||X- WH||_{F}^2 + 0.5 * nu * ||H||_{F}^2 + 0.5 * mul * ||W||_F^2
#          s.t. (1^{T}h_j)^2 = ||hj||^2
#               W >= 0, H >= 0.
#
# @author Wang Shuai
# @date 2018.05.08
###############################################################################

from __future__ import division
from cluster_onmf_manager import ClusterONMFManager # manage the result generated
import numpy as np
from numpy import linalg as LA
import os.path as path
import time
from utils import *

class SNCP1C_Solver(object):
    def __init__(self, data_manager = None, res_dir = None, rank = 4, seed_num = 1, mul = 0, nu = 0):
        if data_manager is None or res_dir is None:
            raise ValueError('Error: some inputs are missing!')
        self.data_manager = data_manager
        self.W, self.H = self.data_manager.gen_inits_WH(init = 'random', seed = seed_num, H_ortho = True)
	self.data_mat = self.data_manager.get_data_mat()
        self.rank = rank
        self.res_dir = res_dir
        self.seed_num = seed_num
	#mul = 1e-2
	self.mul, self.nu = mul, nu
	self.true_labels = self.data_manager.get_labels()
        self.n_factor = LA.norm(self.data_mat, 'fro') ** 2
	self.W_bound = False  # flag to indicate whether to constrain W by upper bound and lower bound
	self.save_acc = False 

	self.time_used = 0 # record the time elapsed when running the simulation
        start_time = time.time()
        self.initialize_penalty_para()
        end_time = time.time()
        self.time_used += end_time - start_time
        self.set_tol(1e-5)
	self.set_max_iters(500)

	W_bound = 'W_bound' if self.W_bound else 'W_nobound'

	# we construct a result manager to manage and save the result
	res_dir1 = path.join(res_dir, 'sncp1c', self.data_manager.get_data_name(), 'cls' + str(rank), W_bound, \
	    'inner' + str(self.inner_tol) + '&gamma' + str(self.gamma) + '&mul' + str(self.mul) + '&nu' + str(self.nu), 'seed' + str(self.seed_num))
        self.res_manager = ClusterONMFManager(root_dir = res_dir1, save_pdv = False) # get an instance of ClusterONMFManager to manage the generated result

    def set_tol(self, tol = 1e-5):
	self.TOL = tol

    def set_max_iters(self, iters = 1000):
	self.max_iters = iters

    def initialize_penalty_para(self):
        self.rho = 1e-8
	self.gamma = 1.1
	self.inner_tol = 1e-3

        (ha, hb) = self.H.shape
        self.I_ha = np.asmatrix(np.eye(ha))
        self.all_1_mat = np.asmatrix(np.ones((ha, ha)))
	
        self.max_val = np.max(self.data_mat) 
	self.min_val = np.min(self.data_mat)

    def get_nmf_cost(self, W, H):
        res = LA.norm(self.data_mat - W * H, 'fro')**2 / self.n_factor
        return res

    def get_obj_val(self, W, H):
	res = LA.norm(self.data_mat - W * H, 'fro')**2 / self.n_factor + \
				0.5 * self.nu * LA.norm(H, 'fro') ** 2 + \
				0.5 * self.mul * LA.norm(W, 'fro') ** 2 
	return res

    def get_penalized_obj(self, W, H):
        '''
        objective function
            1/2 ||X - WH||_{F}^{2} + 0.5 * rho * sum{||hj||_{1}^{2} - ||hj||_{2}^{2}} + 0.5 * nu * ||H||_{F}^2 + 0.5 * mul * ||W||_F^2
        '''
        (ha, hb) = H.shape
        tmp = 0
        for k in range(hb):
            tmp = tmp + (LA.norm(H[:, k], 1)**2 - LA.norm(H[:, k])**2)
        return LA.norm(self.data_mat - W * H, 'fro') ** 2 / self.n_factor + 0.5 * self.rho * tmp + \
					0.5 * self.nu * LA.norm(H, 'fro') ** 2 + 0.5 * self.mul * LA.norm(W, 'fro') ** 2
    def get_onmf_cost(self, W, H, nu = 0, mul = 0):
        ''' This function returns the approximation error of ONMF based on current W and H

        Args:
            W (numpy array or mat): the factor W
            H (numpy array or mat): the factor H
            nu (float): the penalty parameter
        Returns:
            the cost
        '''
        res = LA.norm(self.data_mat - W * H, 'fro')**2 / self.n_factor \
		+ 0.5 * nu * LA.norm(H, 'fro')** 2 \
                + 0.5 * mul * LA.norm(W, 'fro') ** 2
        return res

    def get_sncp_cost(self, W, H, nu = 0, mul = 0, rho = 0):
        ''' This function returns the cost of the penalized subproblem when using SNCP

        Args:
            W (numpy array or mat): the factor W
            H (numpy array or mat): the factor H
            nu (float): the parameter nu * ||H||_F^2
            mul (float): the parameter mul * ||W||_F^2
            rho (float): the penalty parameter rho * \sum_j (||hj||_1^2 - ||hj||_2^2)
        Returns:
            the cost
        '''
        (ha, hb) = H.shape
        all_1_mat = np.asmatrix(np.ones((ha, ha)))
        p_cost = LA.norm(self.data_mat - W * H, 'fro') ** 2 / LA.norm(self.data_mat, 'fro') ** 2 \
                + 0.5 * nu * LA.norm(H, 'fro')** 2  \
                + 0.5 * mul * LA.norm(W, 'fro') ** 2 \
                + 0.5 * rho * (np.trace(H.transpose() * all_1_mat * H) \
                - LA.norm(H, 'fro') ** 2)

        return p_cost

    def update_prim_var_by_PALM(self, W_init = None, H_init = None, max_iter = 1000, tol = 1e-1, verbose = False):
        '''
        This function alternatively updates the primal variables in a Gauss-Seidel fasion.
        Each update is performed using the proximal gradient method
        Input:
            k           ------ the outer iteration number
            W_init      ------ the initialization for W
            H_init      ------ the initialization for H
            max_iter    ------ the max number of iterations for PALM
            tol         ------ the tolerance for stopping PALM
        '''
        if W_init is None or H_init is None:
            raise ValueError('Error: inner iterations by PLAM are lack of initializations!')
	
	(ha, hb) = H_init.shape
	start_time = time.time()
        #H_j_pre, W_j_pre, H_j_cur, W_j_cur = np.asmatrix(np.copy(H_init)), np.asmatrix(np.copy(W_init)), np.asmatrix(np.copy(H_init)), np.asmatrix(np.copy(W_init))
	H_j_pre, W_j_pre = H_init, W_init
	tmp = self.rho * self.all_1_mat + (self.nu - self.rho) * self.I_ha
	end_time = time.time()
        self.time_used += end_time - start_time

        for j in range(max_iter):
            # update H and W by proximal gradient method respectively
	    start_time = time.time()
	    Hessian = 2 * W_j_pre.transpose() * W_j_pre / self.n_factor + tmp
            #egenvals = LA.eigvalsh(Hessian)
            t = 0.51 * LA.eigvalsh(Hessian)[ha - 1]
	    #t = 0.51 * LA.norm(Hessian, 'fro')
	    grad_H_pre = Hessian * H_j_pre - 2 * W_j_pre.transpose() * self.data_mat / self.n_factor
	    #H_j_cur = np.maximum(0, H_j_pre - grad_H_pre / t)
	    H_j_cur = np.maximum(0, H_j_pre - grad_H_pre / t)
	
  
	    Hessian = H_j_cur * H_j_cur.transpose()
            #c = 0.5 * LA.norm(Hessian, 'fro')
            c = 0.51 * LA.eigvalsh(Hessian)[ha - 1]
            grad_W_pre = W_j_pre * Hessian - self.data_mat * H_j_cur.transpose()
            #W_j_cur = np.maximum(0, W_j_pre - grad_W_pre / c)
	    if self.W_bound:
	        W_j_cur = np.minimum(self.max_val, np.maximum(0, W_j_pre - grad_W_pre / c))
	    else:
		W_j_cur = np.maximum(0, W_j_pre - grad_W_pre / c)
 		
	    end_time = time.time()
            self.time_used += end_time - start_time

            # check the convergence
            H_j_change = LA.norm(H_j_cur - H_j_pre, 'fro') / LA.norm(H_j_pre, 'fro')
            W_j_change = LA.norm(W_j_cur - W_j_pre, 'fro') / LA.norm(W_j_pre, 'fro')

            # update the pres
            H_j_pre = np.asmatrix(np.copy(H_j_cur))
            W_j_pre = np.asmatrix(np.copy(W_j_cur))

            if W_j_change + H_j_change < tol:
                break
        
        return (W_j_cur, H_j_cur, j + 1)

    def update_scheme(self):
        '''
        The updating rules for primal variables, W, H and the penalty parameter rho
        use proximal gradient method to update each varialbe once for each iteration
        '''
        (self.W, self.H, inner_iter_num) = self.update_prim_var_by_PALM(self.W, self.H, 3000, self.inner_tol, verbose = False)

	(ha, hb) = self.H.shape
        H_norm = np.asmatrix(np.diag(np.diag(self.H * self.H.transpose()) ** (-0.5))) * self.H
        fea = LA.norm(H_norm * H_norm.transpose() - np.asmatrix(np.eye(ha)), 'fro') / (ha * ha)
            
	start_time = time.time()
	if fea >= 1e-10:
	    self.rho = np.minimum(self.rho * self.gamma,  1e20)
	end_time = time.time()
	self.time_used += end_time - start_time
	
        print 'rho : ' + str(self.rho) + ', nu: ' + str(self.nu) + ', mul: ' + str(self.mul)
	
      	
        return inner_iter_num, self.time_used

    def solve(self):
        '''
        problem formulation
            min 1/2 ||X - WH||_{F}^{2} + 0.5 * rho * sum{||hj||_{1}^{2} - ||hj||_{2}^{2}} + 0.5 * nu * ||H||_F^2 + 0.5 * mul * ||W||_F^2
        '''
        

        print self.H[:, 0]

        fea = 100
	converge = False
	iter_num = 0
	
	self.res_manager.push_W(self.W)  # store W
        self.res_manager.push_H(self.H)  # store H
        self.res_manager.push_H_norm_ortho()  # store feasibility
        self.res_manager.calculate_cluster_quality(self.true_labels) # calculate and store clustering quality
	self.res_manager.push_time(self.time_used)
	
	
        print 'Start to solve the problem by SNCP1 ----------'
        while not converge:

            # update the variable W , H
	    '''
            num = self.update_scheme()
	    '''
	    (self.W, self.H, inner_iter_num) = self.update_prim_var_by_PALM(self.W, self.H, 1000, self.inner_tol, verbose = False)
	    #print 'time used: ' + str(self.time_used)
		            
            #print 'onmf_SNCP1: iter = ' + str(iter_num)
	  
            # store the generated results by result manager
	    self.res_manager.push_W(self.W)  # store W
            self.res_manager.push_H(self.H)  # store H
            self.res_manager.push_H_norm_ortho()  # store feasibility
            self.res_manager.push_W_norm_residual()
            self.res_manager.push_H_norm_residual()
            self.res_manager.calculate_cluster_quality(self.true_labels) # calculate and store clustering quality
            self.res_manager.push_time(self.time_used)
	    
	    iter_num = iter_num + 1
	    
	    fea = self.res_manager.peek_H_norm_ortho()
	    NR = self.res_manager.peek_W_norm_residual() + self.res_manager.peek_H_norm_residual()

	    print 'nr: ' + str(NR)

	    start_time = time.time()
	    if fea >= 1e-20:
	    	self.rho = np.minimum(self.rho * self.gamma,  1e20)
            if iter_num > self.max_iters: converge = True
            elif NR < self.TOL and fea < 1e-5: converge = True
	    else: converge = False
	    end_time = time.time()
	    self.time_used += end_time - start_time

	    print 'rho : ' + str(self.rho) + ', nu: ' + str(self.nu) + ', mul: ' + str(self.mul)

	'''
        print 'HTH:'
        print self.H * self.H.transpose()
	print 'the L2-norm of columns of H:'
        print LA.norm(self.H, axis = 0)
	'''
	print 'time used: ' + str(self.time_used)

        print 'Stop the solve the problem ---------'
	self.res_manager.write_to_csv()  # store the generated results to csv files

    ''' return the solution W, H '''
    def get_solution(self):
        return self.W, self.H


    def get_time(self):
	return self.time_used

    ''' return the cluster assignment from H '''
    def get_cls_assignment_from_H(self):
        labels = np.argmax(np.asarray(self.H), 0)
        if len(labels) != self.data_mat.shape[1]:
            raise ValueError('Error: the size of data samples must = the length of labels!')
        return labels


