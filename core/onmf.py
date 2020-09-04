################################################################################
# The script is to solve the ONMF problem
#          min ||X - WH||_{F}^{2} st. W >= 0, H >= 0, H_{T} H = 0
# It is the child class of NMF.
#
# @author Wang Shuai
# @date 2018.03.30
################################################################################
from __future__ import division
from nmf import NMF
from dtpp_solver import DTPP_Solver
#from sncp_solver import SNCP_Solver
from sncp1_solver import SNCP1_Solver
from sncp1c_solver import SNCP1C_Solver
from sncp2_solver import SNCP2_Solver
from sncp2c_solver import SNCP2C_Solver
from sncp3_solver import SNCP3_Solver
#from sncp4c_solver import SNCP4C_Solver
from hals_solver import HALS_Solver
from onpmf_solver import ONPMF_Solver
from onmf_stf_solver import ONMF_STF_Solver
import numpy as np
import numpy.linalg as LA

class ONMF(NMF):
    def __init__(self, data_manager, res_dir, rank = 4, seed_num = 1, mul = 0, nu = 10):
        super(ONMF, self).__init__(data_manager, res_dir, rank, seed_num)
        self.solver = None
	self.mul = mul
	self.nu = nu

    def solve(self, m_name = 'dtpp'):
        '''
        perform the orthogonality constrained nonnegative matrix factorization
        We have five methods (solvers):
            0: the dtpp method which use multiplicative rules
            1: vs-admm method which combines variable splitting and admm
            2: pamal method
            3. bsum method
            4. penalty_method  with alternating scaled projected gradient method
        '''
        #(W_init, H_init) = self.data_manager.gen_inits_WH(seed = self.seed_num, H_ortho = True)
	# data_mat = self.data_manager.get_data_mat()
        if m_name == 'dtpp':
            solver = DTPP_Solver(self.data_manager, self.res_dir, self.rank, self.seed_num)
        elif m_name == 'sncp1':
            solver = SNCP1_Solver(self.data_manager, self.res_dir, self.rank, self.seed_num, self.mul, self.nu)
	elif m_name == 'sncp1c':
	    solver = SNCP1C_Solver(self.data_manager, self.res_dir, self.rank, self.seed_num, self.mul, self.nu)
	elif m_name == 'sncp2':
	    solver = SNCP2_Solver(self.data_manager, self.res_dir, self.rank, self.seed_num, self.mul, self.nu)
	elif m_name == 'sncp2c':
	    solver = SNCP2C_Solver(self.data_manager, self.res_dir, self.rank, self.seed_num, self.mul, self.nu)
	elif m_name == 'sncp3':
	    solver = SNCP3_Solver(self.data_manager, self.res_dir, self.rank, self.seed_num, self.mul, self.nu)
        elif m_name == 'hals':
            solver = HALS_Solver(self.data_manager, self.res_dir, self.rank, self.seed_num)
        elif m_name == 'onpmf':
            solver = ONPMF_Solver(self.data_manager, self.res_dir, self.rank, self.seed_num)
        elif m_name == 'onmf-stf':
            solver = ONMF_STF_Solver(self.data_manager, self.res_dir, self.rank, self.seed_num)
        else:
            raise ValueError('Error: no other methods to be used for ONMF!')
        if not m_name in {'sncp1', 'sncp1c', 'sncp2c', 'sncp3'} :
            solver.set_tol(1e-5)
	if not m_name in {'onpmf', 'sncp1', 'sncp2', 'sncp3', 'sncp1c', 'sncp2c'}:
	    solver.set_max_iters(2000)
        solver.solve()
        #(self.W, self.H) = solver.get_solution()
        #return solver.get_opt_obj_and_fea(), solver.get_iter_and_time()
	return solver.get_cls_assignment_from_H(), solver.get_time(), solver.get_solution()


