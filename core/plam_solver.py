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
import os.path as path
import time

class PLAM_Solver(object):
    def __init__(self, data_manager = None , res_dir = None, rank = 4, seed_num = 1):
        if None in [data_manager, res_dir]:
            raise ValueError('Error: some inputs are missing!')
        self.data_manager = data_manager
	self.data_mat = self.data_manager.get_data_mat()
        self.W, self.H = self.data_manager.gen_inits_WH(init = 'random', seed = seed_num, H_ortho = False)
        self.rank = rank
        self.res_dir = res_dir
        self.seed_num = seed_num
        self.converge = Convergence(res_dir)
        #np.random.seed(seed_num)  # set the seed so that each run will get the same initial values
        (m, n) = self.data_mat.shape
        self.n_factor = m * n # set the normalization factor to normalize the objective value
        self.time_used = 0 # record the time used by the method
        print 'data_mat'
        print self.data_mat

    def set_max_iters(self, num):
        self.converge.set_max_iters(num)

    def set_tol(self, tol):
        self.converge.set_tolerance(tol)

    def get_obj_val(self):
        res = LA.norm(self.data_mat - self.W * self.H, 'fro')**2 / self.n_factor
        return res

    def get_iter_num(self):
        return self.converge.len()

    def update_prim_var(self, var_name):
        if var_name == 'W':
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

        obj_val = self.get_obj_val()
        print 'The initial error: iter = ' + str(self.get_iter_num()) + ', obj =' + str(obj_val)
        self.converge.add_obj_value(obj_val)
        self.converge.add_prim_value('W', self.W)
        self.converge.add_prim_value('H', self.H)

        print 'Start to solve the problem by PALM ----------'
        while not self.converge.d():
            # update the variable W , H iteratively according to palm method
            start_time = time.time()
            self.update_prim_var('W')
            #obj_val = self.get_obj_val()
            #print 'nmf_PLAM: iter = ' + str(self.get_iter_num()) + ',after update W, obj = ' + str(obj_val)

            self.update_prim_var('H')
            end_time = time.time()
            self.time_used += end_time - start_time

            obj_val = self.get_obj_val()
            print 'nmf_PLAM: iter = ' + str(self.get_iter_num()) + ',after update H, obj = ' + str(obj_val)

            # store the newly obtained values for convergence analysis
            self.converge.add_prim_value('W', self.W)
            self.converge.add_prim_value('H', self.H)

            # store the objective function value
            obj_val = self.get_obj_val()
            self.converge.add_obj_value(obj_val)
            print 'nmf_PLAM: iter = ' + str(self.get_iter_num()) + ', obj = ' + str(obj_val)

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

    ''' simulation result analysis (convergence plot) '''
    def converge_analysis(self):
        # get the dirname to store the result: data file and figure
        dir_name = path.join(self.res_dir, 'onmf', 'plam_k++', 'rank' + str(self.rank), self.data_manager.get_data_name(), 'seed' + str(self.seed_num))
        print 'Start to plot and store the obj convergence ------'
        self.converge.plot_convergence_obj(dir_name)
        print 'Start to plot and store the primal change ------'
        self.converge.plot_convergence_prim_var(dir_name)
        print 'Start to plot and store the fea condition change ------'
        self.converge.plot_convergence_fea_condition(dir_name)
        print 'Start to store the obj values and the factors'
        self.converge.store_obj_val(dir_name)
        self.converge.store_prim_val(-1, dir_name) # store the last element of primal variabl

