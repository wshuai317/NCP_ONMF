################################################################################
# This script is a framework of class for solving the ONMF problme
#
# @author Wang Shuai
# @date 2018.05.01
################################################################################

class ONMF_solver():
    def __init__(self, data_mat, rank):
        self.data_mat = data_mat
        self.rank = rank


