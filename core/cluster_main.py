#############################################################################
# This script to define a main cluster class which can perform clustering
# on input data set (matrix)
#
# @author Wang Shuai
# @date 2018.06.12
#############################################################################

import numpy as np
import pandas as pd
from kmeans import KMeans
from nmf import NMF
from onmf import ONMF
from data_manager import Data_Manager
from sklearn import mixture
import sys
import time
from sklearn.metrics.pairwise import pairwise_distances as p_dist
from nmf_util import consensus_map
import csv
import collections
from cluster_metrics import * 
from utils import save_results_for_KM

class Cluster_Main(object):
    def __init__(self, root_dir = None, data_name = 'syn#1', seed_num = 1, m_name = 'kmeans'):
        self.method_name = m_name
        self.seed_num = seed_num
        self.data_name = data_name
        if root_dir is None:
            raise ValueError('Error: not specify root dir')
        else:
            self.root_dir = root_dir

	data_dir = os.path.join(self.root_dir, 'data')
	self.res_dir = os.path.join(self.root_dir, 'results')

        self.data_manager = self.get_data_manager(root_dir = data_dir, data_name = data_name)

        #print self.data_manager.get_data_mat()
        print self.data_manager.get_labels()

    def get_data_manager(self, root_dir = None, data_name = 'syn#1', seed_num = 1):
        '''
        This function aims to construct a data manager instance to manage the
        data info for the subsequent processing.
        
        '''
	data_info = data_name.split('#')
	data_kind, data_num = data_info[0], int(data_info[1])
	is_real = False if data_kind.startswith('syn') or data_kind.startswith('2d') else True
	has_outliers = True if data_kind.endswith('otlr') else False
		 
        if data_kind == '2d':
	    self.cls_num, num_samples, num_features, dim_reduced = 3, 1000, 2, False
	elif data_kind == 'tcga':
	    self.cls_num, num_samples, num_features, dim_reduced = 33, 11135, 5000, False
	else:
	    self.cls_num, num_samples, num_features, dim_reduced = 10, 1000, 2000, False

        data_manager = Data_Manager(root_dir = root_dir, is_real = is_real, data_kind = data_kind, \
	    data_num = data_num, has_outliers = has_outliers, dim_reduced = dim_reduced, \
	    num_of_features = num_features, num_of_samples = num_samples, num_of_cls = self.cls_num, \
	    seed = seed_num)

        return data_manager

    def solve(self):
        '''
        The function is main entry for clustering. There are several methods to be used
            1. kmeans or Kmeans++
            3. NMF-nonnegative matrix factorization
            4. ONMF-Orthogonality constrained nonnegative matrix factorization
        '''
        if self.method_name in {'kmeans', 'kmeans++', 'kmod', 'msd-km', 'nmf', 'dtpp', 'hals', 'onmf-stf', 'onpmf', 'sncp1c', 'sncp2c', 'sncp4c'}:
	    cls_assign = None
	    time_used = 0
	    if self.method_name == 'kmeans':
	        W, H = self.data_manager.gen_inits_WH(init = 'random', seed = self.seed_num, H_ortho = True)
                initial_centroids = np.asarray(W.transpose())
                start_time = time.time()
                kmeans = KMeans(self.data_manager.get_data_mat(), self.cls_num, self.seed_num)
		print 'initial shape'
		print initial_centroids.shape
                cls_assign,_ = kmeans.solve(initial_centroids, self.data_manager, self.res_dir)
                end_time = time.time()
		time_used = end_time - start_time
	    elif self.method_name == 'kmeans++':
                start_time = time.time()
                kmeans = KMeans(self.data_manager.get_data_mat(), self.cls_num, self.seed_num)
                initial_centroids = kmeans.create_centroids_by_kpp()
                cls_assign,_ = kmeans.solve(initial_centroids)
                end_time = time.time()
		time_used = end_time - start_time
	    elif self.method_name == 'nmf':
                # Before nmf, we should check the validity of input data
                if self.data_manager.contain_zero_rows():
                    raise ValueError('Error: the data matrix has negative values!')
                nmf = NMF(self.data_manager, self.res_dir, self.cls_num, self.seed_num)
                cls_assign, time_used = nmf.solve()

	    elif self.method_name in {'dtpp', 'hals', 'onmf-stf', 'onpmf', 'sncp1c', 'sncp2c', 'sncp4c'}:
	        #if self.data_manager.contain_zero_rows():
                #    raise ValueError('Error: the data matrix has negative values')
		nu = 1e-10
	        mul = 0
		onmf = ONMF(self.data_manager, self.res_dir, self.cls_num, self.seed_num, mul, nu)
                cls_assign, time_used, (W, H) = onmf.solve(self.method_name)
	        

	    # if the dataset is '2d#X', we need to draw a figure to show the clustering 
            # result
            if self.data_name.startswith('2d'):
                dat_path = os.path.join(root_dir, 'results', self.method_name, self.data_name, 'res' + str(self.seed_num) + '.pdf') 
		# get the result directory where the result is stored
                self.data_manager.visualize_data(partition_idx = cls_assign, dat_path = dat_path, data_points = np.asarray(W.transpose()))
		#self.data_manager.visualize_data(partition_idx = cls_assign, dat_path = dat_path)
	    #save clustering performance
	    true_labels = self.data_manager.get_labels()
	    print (true_labels.shape)
            temp_dict = collections.OrderedDict()
            temp_dict['seed'] = self.seed_num
	    temp_dict['time'] = time_used
            temp_dict['Purity'] = calculate_purity(cls_assign, true_labels)
            temp_dict['ARI'] = adjusted_rand_idx = calculate_rand_index(cls_assign, true_labels)
            temp_dict['ACC'] = calculate_accuracy(cls_assign, true_labels)
            temp_dict['NMI'] = calculate_NMI(cls_assign, true_labels)
		
	    return temp_dict

	elif self.method_name in {'sncp', 'sncp1', 'sncp2', 'sncp3'}:

	    for nu in {1e-10}:
		for mul in {0}:
		    onmf = ONMF(self.data_manager, self.res_dir, self.cls_num, self.seed_num, mul, nu)
	    	    #onmf = ONMF(self.data_manager.get_data_mat(), self.res_dir, 20, self.SNR, self.seed_num)
            	    cls_assign, time_used, (W, H) = onmf.solve(self.method_name)

	            if self.data_name.startswith('2d'):
                        dat_path = os.path.join(root_dir, 'results', self.method_name, self.data_name, 'res' + str(self.seed_num) + '.pdf')
			self.data_manager.visualize_data(partition_idx = cls_assign, dat_path = dat_path, data_points = np.asarray(W.transpose()))

	elif self.method_name == 'visualize_data':
	    self.data_manager.visualize_data()

        else:
	    raise ValueError('Error: no other methods are supported now!')
			
 
if __name__ == "__main__":
    import os
    root_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
    print (root_dir)
    if len(sys.argv) < 2:
        print 'no arguments'
    else:
        m_name = sys.argv[1]
        if not m_name in {'nmf', 'kmeans', 'kmeans++', 'kmod', 'dtpp', 'hals', 'onmf-stf', 'onpmf', \
			'sncp', 'sncp1', 'sncp1c', 'sncp2', 'sncp2c', 'sncp3', 'sncp4c', 'visualize_data', 'consensus_map'}:
            print 'wrong method name'
        else:
	    #for d_name in {'syn_otlr#-5', 'syn_otlr#-3', 'syn_otlr#-1', 'syn_otlr#1', 'syn_otlr#3', 'syn_otlr#5'}:
	    #for d_name in {'tdt2#1', 'tdt2#2', 'tdt2#3', 'tdt'tdt2#4', 'tdt2#5', 'tdt2#6', 'tdt2#7'}:
	    #for d_name in {'tdt2#5', 'tdt2#6', 'tdt2#7'}:
	    for d_name in {'syn_otlr#-5'}:
		if d_name.startswith('syn'):
                    num = 2	
		else: num = 10
		if m_name == '':
	            print 'no method'		
		elif m_name == 'visualize_data': # note that only one dataset is allowed at one time when visualizing data !!!!!
		    cluster_proxy = Cluster_Main(root_dir = root_dir, data_name = d_name, seed_num = 0, m_name = m_name)
		    cluster_proxy.solve()
	        elif m_name in {'kmeans', 'kmeans++', 'kmod', 'dtpp', 'hals', 'onmf-stf', 'onpmf', 'nmf', 'sncp1c', 'sncp2c', 'sncp4c'}:
		    res_dict = collections.OrderedDict() # clustering accurary
		    for i in range(1, num+1):	
		    	cluster_proxy = Cluster_Main(root_dir = root_dir, data_name = d_name, seed_num = i, m_name = m_name)
                    	dt = cluster_proxy.solve()
                        res_dict[i - 1] = dt
		    
		    save_results_for_KM(root_dir, res_dict, m_name, d_name)
		else:
                    for i in range(1, num+1):
                        cluster_proxy = Cluster_Main(root_dir = root_dir, data_name = d_name, seed_num = i, m_name = m_name)
                        cluster_proxy.solve()
	    








