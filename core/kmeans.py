#######################################################################################
# This script implements the Kmeans algorithm for clustering data
#
# @author Wang Shuai
# @date 2018.06.05
#######################################################################################

import numpy as np
from numpy import linalg as LA
#import sys
#from sklearn.cluster import KMeans as KM_Learn
import os.path as path
#import random
import os

class KMeans(object):
    def __init__(self, data_mat, rank, seed_num = 1):
        self.data_mat = np.asarray(data_mat.transpose())
        self.rank = rank
        self.seed_num = seed_num
        self.tol = 1e-6
        self.max_iters = 1000
        np.random.seed(seed_num)

        #print 'data_mat:'
        #print self.data_mat
	print 'kmeans'
	print self.data_mat.shape

    def create_centroids_by_kpp(self):
        '''
        Initializes centroids with kmeans++'

        Output:
            centroids   ------ an arrray with shape (num, num_features)
        '''
        print 'using k++ initialization'

        (m, n) = self.data_mat.shape
        initial_centroids = np.ones((self.rank, n)) * (-1)
        ind_list = []
        idx = np.random.choice(m)
        ind_list.append(idx)
        initial_centroids[0, :] = self.data_mat[idx, :]
        while len(ind_list) < self.rank:
            cent = initial_centroids[0:len(ind_list), :]
            D2 = np.array([min([LA.norm(x - c) ** 2 for c in cent]) for x in self.data_mat])
            probs = D2 / D2.sum()
            cumprobs = probs.cumsum()
            r = np.random.random()
            idx = np.where(cumprobs >= r)[0][0]
            ind_list.append(idx)
            initial_centroids[len(ind_list) - 1, :] = self.data_mat[idx, :]
        print ind_list

        return initial_centroids
        #return initial_centroids


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

        distances = np.ones((m, self.rank)) * (-1)

        for centroid_idx in range(self.rank):
            for data_idx in range(m):
                distances[data_idx, centroid_idx] = LA.norm(self.data_mat[data_idx, :] - centroids[centroid_idx, :])

        return distances

    def computeSSE(self, centroids, cluster_assignments):
        '''
        Calculate the summation of distances between each data point to its own centroids which can be used to
        check the stability of Kmeans solution

        Input:
            centroids an array with each row being a centroid
            cluster_assignments   an array containing the cluster ids to which  each data points belongs
        '''
        sse = 0
        nData, _ = self.data_mat.shape
        for i in range(nData):
            cls_id = cluster_assignments[i]
            sse += LA.norm(self.data_mat[i, :] - centroids[cls_id, :])

        return sse


    def compute_iteration(self, old_centroids):
        '''
        Computes an iteration of the k-means algorithm

        Inputs:
            old_centroids, an array with each row being a centroid  at this iteration

        Returns:
            new_centroids, the updated centroids array
            cluster_assignments, an array containing the current cluster assignments for each sample

        '''

        # calculate the distance
        distances = self.get_distances_to_centroids(old_centroids)

        #print 'distances to centroids'
        #print distances


        # find the each data sample's nearest centroid
        cluster_assignments = np.argmin(distances, axis = 1)

        (m, n) = self.data_mat.shape
        if len(cluster_assignments) != m:
            raise ValueError('The assingments array lenght is not right!')

        # compute the new cenroids
        new_centroids = np.zeros((self.rank, n))
        cluster_counts = np.zeros(self.rank)
        for i in range(m): # for each data sample
            new_centroids[cluster_assignments[i], :] += self.data_mat[i, :]
            cluster_counts[cluster_assignments[i]] += 1

        if np.sum(cluster_counts) != m:
            raise ValueError('The number of samples in clusters is not right!')

        for k in range(self.rank):
            if cluster_counts[k] == 0:
		idx = np.random.choice(m, 1)
		print idx
                new_centroids[k, :] = self.data_mat[idx, :]
            else:
                new_centroids[k, :] = new_centroids[k, :] / cluster_counts[k]

        if np.isnan(new_centroids).any():
            print cluster_counts
            #sys.exit(0)
            raise ValueError('Error: new centroids contains nan value')

        return new_centroids, cluster_assignments

    def solve(self, centroids = None, dat_manager = None, res_dir = None):
        '''
        K-means clustering algorithm

        This algorithm partitions a dataset with rows being the data samples into
        K clusters in which each sample belongs to the cluster with the nearest
        centroid.

        Input:
            flag     a flag to indicate use Kmeans from python lirary or my own code
            centroids, default none,  the initial centroids for the clusters
            Defaults to None in which case they are selected randomly

        Returns:
            an array containts the cluster assignments for all data points

        '''

        # initialize k centroids
        if centroids is None:
            raise ValueError('Error: no initial centroids')
            #centroids = self.create_random_centroids(flag = 1)
        '''
        if flag == 0:
            if centroids is None:
                initial = 'random'
            else:
                initial = centroids
            print initial
            km = KM_Learn(n_clusters=self.rank, init = centroids, n_init = 1, random_state = self.seed_num).fit(self.data_mat)
            clusters = km.predict(self.data_mat)
            self.result_analysis(clusters)
            return clusters
        '''
        # Iteratively compute best clusters until they stabilize
        cluster_assignments = None
	
        lastDistance = 1e100
        #print 'start looping'
        #print self.data_mat[1:10, 1:10]
        for iter_num in range(self.max_iters):
	    # put the clustering heatmpa
	    if cluster_assignments is None:
		acc = 1
	    else:
	    	acc = dat_manager.calculate_accuracy(cluster_assignments)
            dat_path = os.path.join(res_dir, 'kmeans', '2d#5', 'res' + str(iter_num) + '_acc(' + str(acc) + ').pdf')
            dat_manager.visualize_data(partition_idx = cluster_assignments, dat_path = dat_path, data_points = centroids)    
            centroids, cluster_assignments = self.compute_iteration(centroids)
            curDistance = self.computeSSE(centroids, cluster_assignments)
            if lastDistance - curDistance < self.tol or (lastDistance - curDistance)/lastDistance < self.tol:
                break

            lastDistance = curDistance
        #print 'end looping'
        return cluster_assignments, centroids











