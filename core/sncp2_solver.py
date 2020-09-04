###############################################################################
# The script is to use a solver based on SNCP method to solve the ONMF problem
#          min 0.5 * ||X- WH||_{F}^2 + 0.5 * nu * ||H||_{F}^2
#          s.t. 1^{T}h_j = ||hj||_{infinity}
#               W >= 0, H >= 0.
#
# @author Wang Shuai
# @date 2018.11.15
###############################################################################

from __future__ import division
from convergence import Convergence
from cluster_onmf_manager import ClusterONMFManager # manage the result generated
import numpy as np
from numpy import linalg as LA
import os.path as path
import time
from utils import *
import math

class SNCP2_Solver(object):
    def __init__(self, data_manager = None, res_dir = None, rank = 4, seed_num = 1, mul = 0, nu = 0):
        if data_manager is None or res_dir is None:
            raise ValueError('Error: some inputs are missing!')
        self.data_manager = data_manager
        self.W, self.H = self.data_manager.gen_inits_WH(init = 'random', seed = seed_num, H_ortho = True)
	self.data_mat = self.data_manager.get_data_mat()
	self.mul = mul
	self.nu = nu
        self.rank = rank
        self.res_dir = res_dir
        self.seed_num = seed_num
        self.converge = Convergence(res_dir) 
	self.true_labels = self.data_manager.get_labels()     
        self.n_factor = LA.norm(self.data_mat, 'fro') ** 2
	self.W_bound = False  # flag to indicate whether to constrain W by upper bound and lower bound
        self.W_step = 0.51
	self.H_step = 0.51

	self.time_used = 0 # record the time elapsed when running the simulation	
	start_time = time.time()
        self.initialize_penalty_para()
	end_time = time.time()
        self.time_used += end_time - start_time
	self.set_tol(1e-3)
	self.set_max_iters(400)

	W_bound = 'W_bound' if self.W_bound else 'W_nobound'
        self.output_dir = path.join(self.res_dir, 'onmf', 'sncp2_W1H1', \
	    W_bound + '_' + 'epsilon' + str(self.inner_tol) + '&gamma' + str(self.gamma) + '&mul' + str(self.mul) + '&nu' + str(self.nu), \
	    'rank' + str(self.rank), self.data_manager.get_data_name(), 'seed' + str(self.seed_num))

	# we construct a result manager to manage and save the result
	res_dir1 = path.join(res_dir, 'onmf', 'sncp2_new', self.data_manager.get_data_name(), 'cls' + str(rank), W_bound + 'W'+ str(self.W_step) + 'H' + str(self.H_step), \
	    'inner' + str(self.inner_tol) + '&gamma' + str(self.gamma) + '&mul' + str(self.mul) + '&nu' + str(self.nu), 'seed' + str(self.seed_num))
        self.res_manager = ClusterONMFManager(root_dir = res_dir1, save_pdv = False) # get an instance of ClusterONMFManager to manage the generated result

	# initialize some variables to store info
        self.acc_iter = []  # record the clustering accuracy for each iteration
        self.time_iter = [] # record the time for each iteration
        self.nmf_cost_iter = [] # record the nmf cost after each iteration
        self.pobj_iter = [] # record the penalized objective value after each iteration
        self.obj_iter = [] # record the objective value for each iteration

    def set_max_iters(self, num):
        self.converge.set_max_iters(num)

    def set_tol(self, tol):
        self.converge.set_tolerance(tol)

    def initialize_penalty_para(self):
        self.rho = 1e-8
	self.gamma = 1.1
	self.inner_tol = 3e-3

        (ha, hb) = self.H.shape
        self.I_ha = np.asmatrix(np.eye(ha))
        self.B = np.zeros_like(self.H)
        self.all_1_mat = np.asmatrix(np.ones((ha, hb)))
        self.max_val = np.max(self.data_mat)
	self.min_val = np.min(self.data_mat)
        

    def get_nmf_cost(self, W, H):
	res = LA.norm(self.data_mat - np.asmatrix(W) * np.asmatrix(H), 'fro')**2 / self.n_factor
	return res
	
    def get_obj_val(self, W, H):
        res = LA.norm(self.data_mat - np.asmatrix(W) * np.asmatrix(H), 'fro')**2 / self.n_factor \
	    + 0.5 * self.nu * LA.norm(H, 'fro') ** 2 + 0.5 * self.mul * LA.norm(W, 'fro') ** 2
        return res

    def get_penalized_obj(self, W, H):
        '''
        objective function
            ||X - WH||_{F}^{2} + self.rho * sum{||hj||_{1} - ||hj||_{infty}} + 0.5 * ||H||_{F}^2
        '''
        (ha, hb) = H.shape
        tmp = 0
        for k in range(hb):
            tmp = tmp + (LA.norm(H[:, k], 1) - LA.norm(H[:, k], np.inf))
        return LA.norm(self.data_mat - W * H, 'fro') ** 2 / self.n_factor + self.rho * tmp \
		+ 0.5 * self.nu * LA.norm(H, 'fro') ** 2 + 0.5 * self.mul * LA.norm(W, 'fro')

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
            rho (float): the penalty parameter rho * \sum_j (||hj||_1 - ||hj||_{\infty})
        Returns:
            the cost
        '''
        (ha, hb) = H.shape
        tmp = 0
        for k in range(hb):
            tmp = tmp + (LA.norm(H[:, k], 1) - LA.norm(H[:, k], np.inf))
        return LA.norm(self.data_mat - W * H, 'fro') ** 2 / self.n_factor + rho * tmp \
		+ 0.5 * nu * LA.norm(H, 'fro') ** 2 + 0.5 * mul * LA.norm(W, 'fro')


    def get_iter_num(self):
        return self.converge.len()



    def update_prim_var_by_PALM0(self, k, W_init = None, H_init = None, max_iter = 1000, tol = 1e-1, verbose = False):
	'''
        This function alternatively updates the primal variables in a Gauss-Seidel fasion.
        The update of H	is performed using the proximal gradient method
	The update of W is performed using the proximal subgradient method
        Input:
            k           ------ the outer iteration number
            W_init      ------ the initialization for W
            H_init      ------ the initialization for H
            max_iter    ------ the max number of iterations for PALM
            tol         ------ the tolerance for stopping PALM
	    verbose     ------ flag to control output debug info
        '''
	if W_init is None or H_init is None:
            raise ValueError('Error: inner iterations by PLAM are lack of initializations!')
	
	start_time = time.time()  # record the start time
	H_j_pre, W_j_pre = np.asmatrix(np.copy(H_init)), np.asmatrix(np.copy(W_init))
        (ha, hb) = H_j_pre.shape
	end_time = time.time()
        self.time_used += end_time - start_time

	for j in range(max_iter):
            # update H and W by proximal gradient method respectively

            start_time = time.time()
	    self.B.fill(0)
            self.B[H_j_pre.argmax(0), np.arange(hb)] = 1
            Hessian = 2 * W_j_pre.transpose() * W_j_pre / self.n_factor + self.nu * self.I_ha	   
            t = self.H_step * LA.eigvalsh(Hessian)[ha - 1]
            grad_H_pre = Hessian * H_j_pre - 2 * W_j_pre.transpose() * self.data_mat / self.n_factor + \
			self.rho * (self.all_1_mat - self.B)
	    H_j_cur = np.maximum(0, H_j_pre - grad_H_pre / t)


	    Hessian = 2 * H_j_cur * H_j_cur.transpose() / self.n_factor + self.mul * self.I_ha
	    c = self.W_step * LA.eigvalsh(Hessian)[ha - 1]
            grad_W_pre = W_j_pre * Hessian - 2 * self.data_mat * H_j_cur.transpose() / self.n_factor
	    if self.W_bound:
	        W_j_cur = np.minimum(self.max_val, np.maximum(0, W_j_pre - grad_W_pre / c))
	    else:
		W_j_cur = np.maximum(0, W_j_pre - grad_W_pre / c)
           
	    if verbose:
	        obj = self.get_obj_val(W_j_cur, H_j_cur)
	        pobj = self.get_penalized_obj(W_j_cur, H_j_cur)

		# store the info
                # calculate the clustering accurary
                pre_labels = np.argmax(np.asarray(H_j_cur), 0)
                if self.labels is None:
                        raise ValueError('Error: no labels!')
                acc = calculate_accuracy(pre_labels, self.labels)
                self.acc_iter.append(acc)
	
		self.obj_iter.append(obj)
		self.pobj_iter.append(pobj)

		cost = self.get_nmf_cost(W_j_cur, H_j_cur)
		self.nmf_cost_iter.append(cost)

		onmf_cost = self.get_onmf_cost(W_j_cur, H_j_cur, self.nu, self.mul)
 	        sncp_cost = self.get_sncp_cost(W_j_cur, H_j_cur, self.nu, self.mul, self.rho)
                self.res_manager.add_cost_value('onmf_cost_palm', onmf_cost) # store obj val
                self.res_manager.add_cost_value('palm_cost', sncp_cost)
		nmf_cost = self.get_onmf_cost(W_j_cur, H_j_cur, 0, 0)
	        self.res_manager.add_cost_value('nmf_cost_palm', nmf_cost)           
	    
            #check the convergence
            H_j_change = LA.norm(H_j_cur - H_j_pre, 'fro') / LA.norm(H_j_pre, 'fro')
            W_j_change = LA.norm(W_j_cur - W_j_pre, 'fro') / LA.norm(W_j_pre, 'fro')

	    #update the pres
            H_j_pre = np.asmatrix(np.copy(H_j_cur))
            W_j_pre = np.asmatrix(np.copy(W_j_cur))

	    end_time = time.time()
            self.time_used += end_time - start_time
            self.time_iter.append(self.time_used)
	    #self.res_manager.push_time(self.time_used)
            #self.res_manager.push_iters(self.rho, j+1)

            # save the info
            if H_j_change + W_j_change < tol:
		self.res_manager.push_iters(self.rho, j+1)
                break

	return (W_j_cur, H_j_cur, j + 1)


    def update_prim_var_by_PALM1(self, k, W_init = None, H_init = None, max_iter = 1000, tol = 1e-1, verbose = False):
        '''
        This function alternatively updates the primal variables in a Gauss-Seidel fasion.
        Each update is performed using the proximal gradient method
        Input:
            k           ------ the outer iteration number
            W_init      ------ the initialization for W
            H_init      ------ the initialization for H
            max_iter    ------ the max number of iterations for PALM
            tol         ------ the tolerance for stopping PALM
	    verbose 	------ flag to control output debug info
        '''
        if W_init is None or H_init is None:
            raise ValueError('Error: inner iterations by PLAM are lack of initializations!')

        start_time = time.time()  # record the start time
        H_j_pre, W_j_pre, H_j_cur, W_j_cur = H_init, W_init, H_init, W_init
	(ha, hb) = H_j_pre.shape
	end_time = time.time() 
	self.time_used += end_time - start_time

        for j in range(max_iter):
            # update H and W by proximal gradient method respectively
	    #if verbose:
            #    print 'PALM1: inner iter = ' + str(j) + ', before H, obj_val = ' + \
            #			str(self.get_obj_val(W_j_pre, H_j_pre)) + ', penalized_obj = ' + str(self.get_penalized_obj(W_j_pre, H_j_pre))

	    start_time = time.time()
	    # keep the infinity norm as a non-smooth part
	    Hessian = 2 * W_j_pre.transpose() * W_j_pre / self.n_factor + self.nu * self.I_ha
	    t = self.H_step *LA.eigvalsh(Hessian)[ha - 1]
	    grad_H_pre = Hessian * H_j_pre - 2 * W_j_pre.transpose() * self.data_mat / self.n_factor + self.rho * self.all_1_mat

	    H_j_cur = H_j_pre - grad_H_pre / t
	    self.B.fill(0)
	    self.B[H_j_cur.argmax(0), np.arange(hb)] = 1
	    H_j_cur += (self.rho / t) * self.B
	    H_j_cur = np.maximum(H_j_cur, 0)

    
	    Hessian = 2 * H_j_cur * H_j_cur.transpose() / self.n_factor + self.mul * self.I_ha
            c = self.W_step * LA.eigvalsh(Hessian)[ha - 1]
            grad_W_pre = W_j_pre * Hessian - 2 * self.data_mat * H_j_cur.transpose() / self.n_factor
	    if self.W_bound:
	        W_j_cur = np.minimum(self.max_val, np.maximum(0, W_j_pre - grad_W_pre / c))
	    else:
		W_j_cur = np.maximum(0, W_j_pre - grad_W_pre / c)

	    if verbose:
		obj = self.get_obj_val(W_j_cur, H_j_cur)
                pobj = self.get_penalized_obj(W_j_cur, H_j_cur)
                # calculate the clustering accurary
                pre_labels = np.argmax(np.asarray(H_j_cur), 0)
                if self.true_labels is None:
                        raise ValueError('Error: no labels!')
                acc = calculate_accuracy(pre_labels, self.true_labels)
                self.acc_iter.append(acc)

                self.obj_iter.append(obj)
                self.pobj_iter.append(pobj)

                cost = self.get_nmf_cost(W_j_cur, H_j_cur)
                self.nmf_cost_iter.append(cost)
                onmf_cost = self.get_onmf_cost(W_j_cur, H_j_cur, self.nu, self.mul)
	        sncp_cost = self.get_sncp_cost(W_j_cur, H_j_cur, self.nu, self.mul, self.rho)
                self.res_manager.add_cost_value('onmf_cost_palm', onmf_cost) # store obj val
                self.res_manager.add_cost_value('palm_cost', sncp_cost)	    
                nmf_cost = self.get_onmf_cost(W_j_cur, H_j_cur, 0, 0)
                self.res_manager.add_cost_value('nmf_cost_palm', nmf_cost)
            
            # check the convergence
            H_j_change = LA.norm(H_j_cur - H_j_pre, 'fro') / LA.norm(H_j_pre, 'fro')
            W_j_change = LA.norm(W_j_cur - W_j_pre, 'fro') / LA.norm(W_j_pre, 'fro')

            # update the pres
            H_j_pre = np.asmatrix(np.copy(H_j_cur))
            W_j_pre = np.asmatrix(np.copy(W_j_cur))

	    end_time = time.time()
            self.time_used += end_time - start_time
            self.time_iter.append(self.time_used)
	    #self.res_manager.push_time(self.time_used)
            #self.res_manager.push_iters(self.rho, j+1)

            if H_j_change + W_j_change < tol:
		self.res_manager.push_iters(self.rho, j+1)
                break
	    

        return (W_j_cur, H_j_cur, j + 1)

    def update_scheme(self):
        '''
        The updating rules for primal variables, W, H and the penalty parameter rho
        use proximal gradient method to update each varialbe once for each iteration
        '''
        # update 
        # (self.W, self.H, inner_iter_num) = self.update_prim_var_by_PALM0(self.get_iter_num(), self.W, self.H, 3000, self.inner_tol, verbose = False)
	(self.W, self.H, inner_iter_num) = self.update_prim_var_by_PALM1(self.get_iter_num(), self.W, self.H, 3000, self.inner_tol, verbose = False)

	# show the feasibility satisfaction level HH^{T} - I
	(ha, hb) = self.H.shape
        H_norm = np.asmatrix(np.diag(np.diag(self.H * self.H.transpose()) ** (-0.5))) * self.H
        fea = LA.norm(H_norm * H_norm.transpose() - np.asmatrix(np.eye(ha)), 'fro') / (ha * ha)
        	
	start_time = time.time()
        #if self.get_iter_num() > 0 and fea > 1e-10:
        self.rho = np.minimum(self.rho * self.gamma, 1e10)
	print self.rho
	end_time = time.time()
        self.time_used += end_time - start_time
       

        return inner_iter_num

    def solve(self):

        '''
        problem formulation
            min ||X - WH||_{F}^{2} + rho * sum{||hj||_{1} - ||hj||_{infty} + 0.5 *nu * ||H||_F^2 * 0.5 * mul * ||W||_F^2
        '''
        obj = self.get_obj_val(self.W, self.H)
	p_obj = self.get_penalized_obj(self.W, self.H)
        print 'The initial error: iter = ' + str(self.get_iter_num()) + ', obj_val =' + str(obj) + ', penalized_obj =' + str(p_obj)
        #self.converge.add_obj_value(cost)
        self.converge.add_obj_value(obj)
        self.converge.add_prim_value('W', self.W)
        self.converge.add_prim_value('H', self.H)

        print self.H[:, 0]
        
	inner_iter_nums = []  # record the inner iterations number
	acc_sncp = []  # record the clustering accuracy for each SNCP iteration
        time_sncp = [] # record the time used after each SNCP iteration
        nmf_cost_sncp = [] # record the nmf cost after each SNCP iteration
        pobj_sncp = [] # record the penalized objective value after each iteration

	cost = self.get_nmf_cost(self.W, self.H)
	nmf_cost_sncp.append(cost)
	pobj_sncp.append(p_obj)

	self.pobj_iter.append(p_obj)
	self.nmf_cost_iter.append(cost)
	self.obj_iter.append(obj)

	# calculate the clustering accurary
        pre_labels = np.argmax(np.asarray(self.H), 0)
        if self.true_labels is None:
	    raise ValueError('Error: no labels!')
	print len(self.true_labels)
        acc = calculate_accuracy(pre_labels, self.true_labels)
        acc_sncp.append(acc)
	self.acc_iter.append(acc)

	time_sncp.append(self.time_used)
	self.time_iter.append(self.time_used)
       
        fea = 100

	self.res_manager.push_W(self.W)  # store W
        self.res_manager.push_H(self.H)  # store H
        self.res_manager.push_H_norm_ortho()  # store feasibility
	nmf_cost = self.get_onmf_cost(self.W, self.H, 0, 0)
	onmf_cost = self.get_onmf_cost(self.W, self.H, self.nu, self.mul)
	sncp_cost = self.get_sncp_cost(self.W, self.H, self.nu, self.mul, self.rho)
        self.res_manager.add_cost_value('onmf_cost_sncp', onmf_cost) # store obj val
        self.res_manager.add_cost_value('sncp_cost', sncp_cost)
        self.res_manager.add_cost_value('onmf_cost_palm', onmf_cost) # store obj val
        self.res_manager.add_cost_value('palm_cost', sncp_cost)
	self.res_manager.add_cost_value('nmf_cost_sncp', nmf_cost)
	self.res_manager.add_cost_value('nmf_cost_palm', nmf_cost)
        cls_assign = self.res_manager.calculate_cluster_quality(self.true_labels) # calculate and store clustering quality
	self.res_manager.push_time(self.time_used)
	
        print 'Start to solve the problem by SNCP2 ----------'
        while not self.converge.d() or fea > 1e-10:

            # update the variable W , H
            num = self.update_scheme()
            inner_iter_nums.append(num)
	    time_sncp.append(self.time_used)
	    print 'time used: ' + str(self.time_used) + ', inner_num: ' + str(num)
            
	    # calculate the clustering accurary
            pre_labels = np.argmax(np.asarray(self.H), 0)
            if self.true_labels is None:
                raise ValueError('Error: no labels!')
            acc = calculate_accuracy(pre_labels, self.true_labels)
            acc_sncp.append(acc)
            
            # store the newly obtained values for convergence analysis
            self.converge.add_prim_value('W', self.W)
            self.converge.add_prim_value('H', self.H)

            # store the nmf approximation error value
            obj = self.get_obj_val(self.W, self.H)
            p_obj = self.get_penalized_obj(self.W, self.H)
	    nmf_cost_sncp.append(self.get_nmf_cost(self.W, self.H))
	   
            self.converge.add_obj_value(obj)
	    pobj_sncp.append(p_obj)
            print 'onmf_SNCP2: iter = ' + str(self.get_iter_num()) + ', obj_val = ' + str(obj) + ' penalized_obj = ' + str(p_obj)

            # store the satisfaction of feasible conditions
            (ha, hb) = self.H.shape
            H_norm = np.asmatrix(np.diag(np.diag(self.H * self.H.transpose()) ** (-0.5))) * self.H
            fea = LA.norm(H_norm * H_norm.transpose() - np.asmatrix(np.eye(ha)), 'fro') / (ha * ha)

	    #print 'normalized orthogonality: ' + str(fea)
            self.converge.add_fea_condition_value('HTH_I', fea)

            # store the generated results by result manager
	    self.res_manager.push_W(self.W)  # store W
            self.res_manager.push_H(self.H)  # store H
            self.res_manager.push_H_norm_ortho()  # store feasibility
            self.res_manager.push_W_norm_residual()
            self.res_manager.push_H_norm_residual()
	    nmf_cost = self.get_onmf_cost(self.W, self.H, 0, 0)
	    onmf_cost = self.get_onmf_cost(self.W, self.H, self.nu, self.mul)
	    sncp_cost = self.get_sncp_cost(self.W, self.H, self.nu, self.mul, self.rho)
            self.res_manager.add_cost_value('onmf_cost_sncp', onmf_cost) # store obj val
            self.res_manager.add_cost_value('sncp_cost', sncp_cost)
            self.res_manager.add_cost_value('onmf_cost_palm', onmf_cost) # store obj val
            self.res_manager.add_cost_value('palm_cost', sncp_cost)
	    self.res_manager.add_cost_value('nmf_cost_sncp', nmf_cost)
            self.res_manager.add_cost_value('nmf_cost_palm', nmf_cost)
            cls_assign = self.res_manager.calculate_cluster_quality(self.true_labels) # calculate and store clustering quality
            self.res_manager.push_time(self.time_used)

        print 'HTH:'
        print self.H * self.H.transpose()
	print 'the L2-norm of columns of H:'
        print LA.norm(self.H, axis = 0)

        
        # show the number of inner iterations
        self.converge.save_data(inner_iter_nums, self.output_dir, 'inner_nums.csv')
	#self.converge.save_data(time_sncp, self.output_dir, 'time_sncp.csv')
	self.converge.save_data(acc_sncp, self.output_dir, 'acc_sncp.csv')
	self.converge.save_data(nmf_cost_sncp, self.output_dir, 'nmf_cost_sncp.csv')
	self.converge.save_data(pobj_sncp, self.output_dir, 'pobj_sncp.csv')

        
        self.converge.save_data(self.obj_iter, self.output_dir, 'obj_iters.csv')
        self.converge.save_data(self.acc_iter, self.output_dir, 'acc_iters.csv')
        self.converge.save_data(self.nmf_cost_iter, self.output_dir, 'nmf_cost_iters.csv')
        self.converge.save_data(self.pobj_iter, self.output_dir, 'pobj_iters.csv')

	self.converge.save_data(self.time_iter, self.output_dir, 'time_iters.csv')
        
        print 'Stop the solve the problem ---------'
        self.converge_analysis()
	self.res_manager.write_to_csv()  # store the generated results to csv files

    ''' return the solution W, H '''
    def get_solution(self):
        return self.W, self.H

    ''' return the optimal obj val '''
    def get_opt_obj_and_fea(self):
        return self.get_nmf_cost(self.W, self.H), self.converge.get_last_fea_condition_value('HTH_I')

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

