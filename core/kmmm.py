#######################################################################################
# This script implements the Kmeans-- algorithm for clustering data with noise
# Please refer to the following paper:
#  Chawla, S. and A. Gionis (2013). k-means--: A unified approach to clustering and outlier detection. SIAM International Conference on Data Mining (SDM13).
#
# @author Wang Shuai
# @date 2020.05.11
#######################################################################################

import numpy as np
from numpy import linalg as LA
#import sys
#from sklearn.cluster import KMeans as KM_Learn
import os.path as path
#import random
from sklearn.metrics.pairwise import euclidean_distances


class KMod(object):
    def __init__(self, data_mat, cls_num, outlier_num):
        self.data_mat = np.asarray(data_mat.transpose())
        self.cls_num = cls_num
	self.outlier_num = int(outlier_num)
        #self.seed_num = seed_num
        self.tol = 1e-10
        self.max_iters = 1000

    def solve(self, seed_num = 0):
        '''
        K-means-- clustering algorithm

        Input:
            seed_num - float the random generator to generate centroids

        Returns:
            an array containts the cluster assignments for all data points

        '''
	num_data, num_feature = self.data_mat.shape
	if num_data < self.cls_num + self.outlier_num:
	    raise ValueError('Error: clusters + outliers > datapoints!')

	print ('Beginning K-means-- clustering--')
        # create k centroids by randomly sampling k points
	np.random.seed(seed_num)
	center_idx = np.random.choice(range(num_data), self.cls_num, replace = False)
	C_current = np.copy(self.data_mat[center_idx, :])
	# initialize iteration count
	iter_idx = 0
	# track convergence
  	conv_delta = 0

	while not self.converged(iter_idx, conv_delta):

	    # track centroids
            # C_prev2= C_prev
            C_prev = np.copy(C_current)

	    dist_to_cls, cls_idx = self.dist_sqr_XC(C_prev)
	    
            # reorder the data points according dist_to_cls (descending order)
	    Q_reorder = np.argsort(-dist_to_cls)
		
	    # consider the first l data points as outliers and remove them temporily
	    QXnet = Q_reorder[self.outlier_num:]
	    cls_idx[Q_reorder[0:self.outlier_num]] = -1
	    # calculate the new centroids
    	    # by averaging the values in Xnet assigned to each Ci-1
	    C_current.fill(0)
	    missing_centroids = []	
	    cls_sizes = np.zeros(self.cls_num)
	    for k in range(self.cls_num): # for each cluster
		#print cls_idx.shape
		k_idx = np.where(cls_idx == k)[0]
		cls_sizes[k] = len(k_idx)
	        if len(k_idx) <= 0: # no data points assigned to cluster k
		   missing_centroids.append(k)
		else:
		   #print k_idx
	           data = self.data_mat[k_idx, :]
		   #print data.shape
		   C_current[k, :] = np.mean(data, axis = 0)

	    if len(missing_centroids) > 0: # there are some clusters without data points
		# choose the data points with largest distance in the biggest cluster
		c_max = np.argmax(cls_sizes)
		if cls_sizes[c_max] <= len(missing_centroids):
		    raise ValueError('Error: the size of largest clusters <= number of missing centroids!')
		c_max_idx = np.where(cls_idx == c_max)[0]
		dist_to_max = dist_to_cls[c_max_idx]
		c_max_idx_reorder = c_max_idx[np.argsort(-dist_to_max)]
		for j in range(len(missing_centroids)): # for each missing centroids
		    C_current[missing_centroids[j], :] = self.data_mat[c_max_idx_reorder[j], :]


	    iter_idx = iter_idx + 1
	    conv_delta = LA.norm(C_current - C_prev, 'fro') / LA.norm(C_prev, 'fro')

	print 'iter_idx : ' + str(iter_idx)

	# get the clustering assignments
	dist_to_cls, cls_assign = self.dist_sqr_XC(C_current)
	return cls_assign


    def converged(self, iter_idx, conv_error):
        '''
        This function check whether the algorithm converges or not

        Args:
            iter_idx - int- current iteration index
            conv_error - float - metric
        Returns:
            True or False
        '''
        if iter_idx <= 0: return False
        elif iter_idx >= self.max_iters: return True
        elif conv_error <= self.tol: return True
        else: return False

    def get_distances_to_centroids(self, centroids):
        '''
        Calculate the distances from each data points to each centroid

        Input:
            cetroids  an array with each row being a centroid with the dimensionarlity
            same to that of data sample

        Returns
            An array with a row for each sample in the dataset and
            a column for the distance to each centroid
        '''

        (m, n) = self.data_mat.shape

        distances = np.ones((m, self.cls_num)) * (-1)

        for centroid_idx in range(self.cls_num):
            for data_idx in range(m):
                distances[data_idx, centroid_idx] = LA.norm(self.data_mat[data_idx, :] - centroids[centroid_idx, :])

        return distances



    def dist_sqr_XC(self, centers):
	''' 
	This function calculate the distance between each data point and its nearest cluster center

	Args:
	    centers -- (numpy array) --- the centers
	Returns:
	    an array of distances
	'''
	#dist = euclidean_distances(self.data_mat, centers)** 2
	dist = self.get_distances_to_centroids(centers)
	dist_to_cls = np.min(dist, axis = 1)
	cls_idx = np.argmin(dist, axis = 1)
	return dist_to_cls, cls_idx











