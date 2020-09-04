####################################################################################
# The script is to use a solver based on the NMF multiplicative rule to solve the
# NMF problem
#
# @author Wang Shuai
# @date 2018.06.04
###################################################################################
from __future__ import division
from convergence import Convergence
import numpy as np
from numpy import linalg as LA
import os.path as path
import time

class NMF_Solver(object):
    def __init__(self, data_manager = None, res_dir = None, rank = 4, seed_num = 1):
        if data_manager is None or res_dir is None:
            raise ValueError('Error: some inputs are missing!')
        self.data_manager = data_manager
	self.data_mat = self.data_manager.get_data_mat()
        self.W, self.H = self.data_manager.gen_inits_WH(init = 'random', seed = seed_num, H_ortho = False)
        self.res_dir = res_dir
        self.rank = rank
        self.seed_num = seed_num
        self.converge = Convergence(res_dir)
        #np.random.seed(seed_num)   # set the seed so that each run will get the same initial values
        (m, n) = self.data_mat.shape
        #self.n_factor = m * n # set the normalization factor to normalize the objective value
        self.n_factor = LA.norm(self.data_mat, 'fro') ** 2
        self.flag = 0 # a flag to indicate which (problem, method) pair to be used,
                      # 0: nmf_fro + multiplicative rule
                      # 1: nmf_kl + nultiplicative rule
                      # 2: nmf_fro + palm
        self.time_used = 0 # record the time used by the method

    def set_max_iters(self, num):
        self.converge.set_max_iters(num)

    def set_tol(self, tol):
        self.converge.set_tolerance(tol)

    def get_obj_val(self, flag = 0):
        if flag == 0 or flag == 2:
            res = LA.norm(self.data_mat - self.W * self.H, 'fro')**2 / self.n_factor
        else:
            # initial reconstruction
            R = self.W * self.H
            # compute KL-divergence
            #errs(k) = sum(V(:) .* log(V(:) ./ R(:)) - V(:) + R(:));
            tmp = np.multiply(self.data_mat.flatten('F'), np.log(np.divide(self.data_mat.flatten('F'), R.flatten('F'))))
            res = np.sum(tmp - self.data_mat.flatten('F') + R.flatten('F')) / np.sqrt(self.n_factor)
        return res

    def get_iter_num(self):
        return self.converge.len()

    def update_prim_var(self, var_name, flag = 0):
        if flag == 1:
            #preallocate matrix of ones
            (m, n) = self.data_mat.shape
            Onm  = np.asmatrix(np.ones((m, n)))

        if var_name == 'W':
            if flag == 0:
                #W = W .* ((V * H') ./ max(W * (H * H'), myeps));
                temp = np.divide(self.data_mat * self.H.transpose(), \
                        np.maximum(self.W * (self.H * self.H.transpose()), 1e-20))
                self.W = np.multiply(self.W, temp)
            elif flag == 1:
                # initial reconstruction
                R = self.W * self.H
                #W = W .* (((V ./ R) * H') ./ max(Onm * H', myeps));
                temp = np.divide(np.divide(self.data_mat, R) * self.H.transpose(), \
                        np.maximum(Onm * self.H.transpose(), 1e-20))
                self.W = np.multiply(self.W, temp)
            else:
                step_size = 1
                beta = 0.5
                gx = LA.norm(self.data_mat - self.W * self.H, 'fro') ** 2
                gradient = (self.W * self.H - self.data_mat) * self.H.transpose()
                while True:
                    tmp = self.W - step_size * gradient
                    fz = LA.norm(self.data_mat - tmp * self.H, 'fro') ** 2
                    if fz <= gx + 0.5 * np.trace(gradient.transpose() * (tmp - self.W)):
                        self.W = tmp
                        break
                    step_size = step_size * beta
                self.W = np.maximum(self.W, 0)
        elif var_name == 'H':
            if flag == 0:
                #H = H .* ( (W'* V) ./ max((W' * W) * H, myeps))
                temp = np.divide(self.W.transpose() * self.data_mat, \
                        np.maximum(self.W.transpose() * self.W * self.H, 1e-20))
                self.H = np.multiply(self.H, temp)
            elif flag == 1:
                # initial reconstruction
                R = self.W * self.H
                #H = H .* ((W' * (V ./ R)) ./ max(W' * Onm, myeps));
                temp = np.divide(self.W.transpose() * np.divide(self.data_mat, R), \
                        np.maximum(self.W.transpose() * Onm, 1e-20))
                self.H = np.multiply(self.H, temp)
            else:
                step_size = 1
                beta = 0.5
                gx = LA.norm(self.data_mat - self.W * self.H, 'fro') ** 2
                gradient = self.W.transpose() * (self.W * self.H - self.data_mat)
                while True:
                    tmp = self.H - step_size * gradient
                    fz = LA.norm(self.data_mat - self.W * tmp, 'fro') ** 2
                    if fz <= gx + 0.5 * np.trace(gradient.transpose() * (tmp - self.H)):
                        self.H = tmp
                        break
                    step_size = beta * step_size
                self.H = np.maximum(self.H, 0)
        else:
            raise ValueError('Error: no other variable should be updated!')

    def solve(self):

        obj_val = self.get_obj_val(self.flag)
        print 'The initial error: iter = ' + str(self.get_iter_num()) + ', obj =' + str(obj_val)
        self.converge.add_obj_value(obj_val)
        self.converge.add_prim_value('W', self.W)
        self.converge.add_prim_value('H', self.H)

        print 'Start the solve the problem by NMF ----------'
        while not self.converge.d():
            # update the variable W, H iteratively according to NMF multiplicative rules
            start_time = time.time()   # record the start time
            self.update_prim_var('W', self.flag)
            self.update_prim_var('H', self.flag)
            end_time = time.time()    # record the end time
            self.time_used += end_time - start_time

            # store the newly obtained values for convergence analysis
            self.converge.add_prim_value('W', self.W)
            self.converge.add_prim_value('H', self.H)

            # store the objective function value
            obj_val = self.get_obj_val(self.flag)
            self.converge.add_obj_value(obj_val)
            print 'NMF solver: iter = ' + str(self.get_iter_num()) + ', obj = ' + str(obj_val)

            # store the satisfaction of feasible conditions
            (ha, hb) = self.H.shape
            fea = LA.norm(self.H * self.H.transpose() - np.asmatrix(np.eye(ha)), 'fro') / (ha * ha)
            self.converge.add_fea_condition_value('HTH_I', fea)


        print 'Stop to solve the problem ----------'
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
        if self.flag == 0:
            sub_folder = 'nmf_fro_mul'
        elif self.flag == 1:
            sub_folder = 'nmf_kl'
        else:
            sub_folder = 'nmf_fro_palm'

        dir_name = path.join(self.res_dir, 'nmf', sub_folder, 'rank' + str(self.rank), self.data_manager.get_data_name(), 'seed' + str(self.seed_num))
        print 'Start to plot and store the obj convergence ------'
        self.converge.plot_convergence_obj(dir_name)
        print 'Start to plot and store the primal change ------'
        self.converge.plot_convergence_prim_var(dir_name)
        print 'Start to plot and store the fea condition change ------'
        self.converge.plot_convergence_fea_condition(dir_name)
        print 'Start to store the obj values and the factors'
        self.converge.store_obj_val(dir_name)
        self.converge.store_prim_val(-1, dir_name) # store the last element of primal variabl







