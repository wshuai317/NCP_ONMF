#######################################################################################
# This script implements the Kmeans algorithm for clustering data
#
# @author Wang Shuai
# @date 2018.06.05
#######################################################################################

import numpy as np
from numpy import linalg as LA
import sys
from sklearn.cluster import KMeans as KM_Learn
import os.path as path
import itertools


class KMeans(object):
    def __init__(self, data_mat, res_dir, rank, seed_num = 1):
        self.data_mat = np.asarray(data_mat.transpose())
        self.res_dir = res_dir
        self.rank = rank
        self.seed_num = seed_num
        self.tol = 1e-6
        self.max_iters = 1000
        np.random.seed(seed_num)

    def rand_init_centroids(self, cluster_atts_idxs):
        centroids = list(itertools.repeat(None, self.rank))
        (num, dim) = self.data_mat.shape

        idxs = list(range(0, num))

        for i in range(0, self.rank):
            r = np.random.randint(0, length(idxs))
            r_idx = idxs[r]
            del idxs[r]
            centroids[i] = self.data_mat[r_dix, :]
            #datum = data_rows[r_idx]
            #centroids[i] = project_cluster_atts(datum, cluster_atts_idxs)
        return centroids

    #def project_cluster_atts(datum, cluster_atts_idxs):
    #    return [datum[x] for x in cluster_atts_idxs]

    def find_closest_centroid(self, centroids, datum):
        closest_centroid_idx = None
        closest_centroid_distance = None
        for centroid_idx in range(0, len(centroids)):
            centroid_datum = centroids[centroid_idx]

            if None is centroid_datum:
                continue

            distance = self.distance_between(datum, centroid_datum)

            if closest_centroid_distance is None or distance < closest_centroid_distance:
                closest_centroid_distance = distance
                closest_centroid_idx = centroid_idx

        return closest_centroid_distance, closest_centroid_idx

    def assignment_step(centroids, cluster_atts_idx, data_rows):
        cluster_assignment = {}
        #distortion = 0
        (num, dim) = self.data_mat.shape

        for datum_idx in range(0, num):
            datum = self.data_mat[datum_idx, :]

            closest_centroid_distance, closest_centroid_idx = find_closest_centroid(centroids, cluster_atts_idx, datum)

            if closest_centroid_idx not in cluster_assignment:
                cluster_assignment[closest_centroid_idx] = list()
            cluster_assignment[closest_centroid_idx].append(datum_idx)
            #distortion += closest_centroid_distance

        return cluster_assignment

    def distance_between(self, datum, centroid_datum):
        s = 0
        for i in range(0, len(datum)):
            centroid_datum_att_value = datum[i]
            datum_att_value = centroid_datum[i]

            s += math.pow(abs(centroid_datum_att_value - datum_att_value), 2)

        return math.sqrt(s)

    def update_centroids(self, cluster_assignments):
        centroids = list(itertools.repeat(None, self.rank))

        num_of_atts = len(cluster_atts_idxs)

        for cluster_id in sorted(cluster_assignments):
            data_for_cluster_idxs = cluster_assignments[cluster_id]
            new_centroid = list(itertools.repeat(0.0, num_of_atts))

            num_in_cluster = len(data_for_cluster_idxs)

            for data_for_cluster_idx in data_for_cluster_idxs:
                datum = data_rows[data_for_cluster_idx]
                datum_comparable_atts = project_cluster_atts(datum, cluster_atts_idxs)

                for cluster_atts_idx_idx in range(0, num_of_atts):
                    new_centroid[cluster_atts_idx_idx] += datum_comparable_atts[cluster_atts_idx_idx]

            for cluster_atts_idx_idx in range(0, num_of_atts):
                new_centroid[cluster_atts_idx_idx] /= num_in_cluster
                centroids[cluster_id] = new_centroid

        return centroids

    def kmeans(self, k, cluster_atts, cluster_atts_idxs, centroids = None):
        # select initial centroids
        #data_rows = data['rows']

        #centroids = init_func(data_rows, k, cluster_atts_idxs)
        if centroids is None:
            centroids = self.rand_init_centroids()

        cluster_assignments, distortion = self.assignment_step(centroids, cluster_atts_idxs, data_rows)

        #plot_cluster_assignments(cluster_assignments, centroids, data_rows, cluster_atts, cluster_atts_idxs, distortion, plot_config)

        while True:
            centroids = self.update_centroids(data_rows, cluster_assignments, cluster_atts_idxs, k)
            next_cluster_assignments, distortion = assignment_step(centroids, cluster_atts_idxs, data_rows)
            if cluster_assignments == next_cluster_assignments:
                break
            cluster_assignments = next_cluster_assignments
            #plot_cluster_assignments(cluster_assignments, centroids, data_rows, cluster_atts, cluster_atts_idxs, distortion, plot_config)
            #plot_cluster_assignments(cluster_assignments, centroids, data_rows, cluster_atts, cluster_atts_idxs, distortion, plot_config)

        return cluster_assignments, centroids, distortion


    def solve(self, flag = 0, centroids = None):
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
            centroids = self.create_random_centroids(flag = 1)
        print 'initial_centroids'
        print centroids

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

        # Iteratively compute best clusters until they stabilize
        cluster_assignments = None

        lastDistance = 1e100

        for iter in range(self.max_iters):
            centroids, cluster_assignments = self.compute_iteration(centroids)
            print 'new assignments'
            print cluster_assignments
            print 'new centroids'
            print centroids
            print 'new iteration'

            curDistance = self.computeSSE(centroids, cluster_assignments)
            if lastDistance - curDistance < self.tol or (lastDistance - curDistance)/lastDistance < self.tol:
                print "# of iterations:", iter
                print "SSE = ", curDistance
                break

        self.result_analysis(cluster_assignments)

        return cluster_assignments


    def result_analysis(self, assignments):
        '''
        transform the obtained cluster assginments to an array with dimension K * m (K: the cluster num, m: the
        data points number).
        '''
        (m, n) = self.data_mat.shape
        H = np.zeros((m, self.rank))
        #print m, n
        #print assignments
        for j in range(m):
            H[j, assignments[j]] = 1

        # get the dirname to store the result: data file and figure
        dir_name = path.join(self.res_dir, 'onmf', 'kmeans', 'rank' + str(self.rank), 'seed_W' + str(self.seed_num))

        real_path = path.join(dir_name, 'H.csv')
        print real_path
        np.savetxt(real_path, np.asmatrix(H), delimiter = ",")










