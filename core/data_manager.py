#######################################################################
# This class are used to manage and process the data before processing
# including importing data from data files, checking the validity of
# data, recording some properties of data
#
#
# @author Wang Shuai
# @date 2018.08.17
#######################################################################
from __future__ import division
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from numpy import linalg as LA
#import itertools
from sklearn.cluster import KMeans
from sklearn.utils import linear_assignment_
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from numpy import linalg as LA
from filemanager import FileManager
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances as pdist
import time
from sklearn import manifold

#import random

class Data_Manager(object):
    def __init__(self, root_dir, is_real, data_kind, data_num, has_outliers = True, dim_reduced = False, \
		num_of_features = 2000, num_of_samples = 1000, num_of_cls = 10, seed = 0):
       	self.root_dir = root_dir
	self.is_real = is_real
	self.num_of_cls = num_of_cls
	self.data_kind = data_kind
	self.data_num = data_num
	self.has_outliers = has_outliers
	self.num_of_features = num_of_features
	self.num_of_samples = num_of_samples
	self.dim_reduced = dim_reduced
	dr_str = 'DR' if dim_reduced else ''
	outliers = 'otlrs' if has_outliers else ''
	if is_real: # save the newly generated data so that we don't need to regenerate it again
	    if not self.data_kind in {'mnist', 'tdt2', 'tcga'}:
		raise ValueError('Error: other data kinds are not supported now!')
	    data_path = os.path.join(self.root_dir, 'real_data', self.data_kind, 'data' + dr_str + '#' + str(self.data_num) + '.csv')
            label_path = os.path.join(self.root_dir, 'real_data', self.data_kind, 'label#' + str(self.data_num) + '.csv')
	    print data_path
	    print label_path
	    print self.root_dir
            if os.path.exists(data_path):# data file exists, just read it
		self.data_mat = self.read_data_from_csvfile(data_path)
		if not self.dim_reduced:
		    if self.data_kind in {'tdt2', 'mnist'}: 
		    	self.data_mat = self.data_mat.transpose()
		self.true_labels = self.read_data_from_csvfile(label_path)
		print 'labels shape: ' + str(self.true_labels.shape)
		if self.data_kind in {'tdt2', 'mnist'}:
		    self.true_labels = self.true_labels.transpose()
		self.true_labels = self.true_labels[0, :] # since labels are stored as matrix, we just extrac row 0
		self.existed = True
	    else: # in case the original dataset without dimension reduction exists
		print False
            	orig_data_path = os.path.join(self.root_dir, 'real_data', self.data_kind, 'data#' + str(self.data_num) + '_seed' + str(seed) + '.csv')
            	orig_label_path = os.path.join(self.root_dir, 'real_data', self.data_kind, 'label#' + str(self.data_num) + '_seed' + str(seed) + '.csv')
		if os.path.exists(orig_data_path):
                    data_mat = self.read_data_from_csvfile(orig_data_path)
                    #self.data_mat = self.data_mat.transpose()[:, 0:20001] # just for testing
                    labels = self.read_data_from_csvfile(orig_label_path)
                    print (data_mat.shape)
                    labels = labels.transpose()[0, :]
		    if self.dim_reduced:
                    	self.data_mat = self.dim_reduction_by_spectral()
                    	self.data_mat = self.data_mat.transpose()
		    f_manager = FileManager(self.root_dir)
                    f_manager.add_file(data_path)
                    np.savetxt(data_path, np.asmatrix(self.data), delimiter = ',')
                    f_manager.add_file(label_path)
                    np.savetxt(label_path, np.asmatrix(self.true_labels), delimiter = ',')
                    self.existed = False
		else:
		    raise ValueError('Error: no available datasets')
	else:
	    print ('seed: ' + str(seed))
	    np.random.seed(seed) # set the seed
	    # at first, we check whether the data file has been generated or not
	    data_path = os.path.join(self.root_dir, 'synthetic_data', self.data_kind + '#' + str(self.data_num) + '_' + dr_str \
                + '_' + str(self.num_of_features) + 'x' + str(self.num_of_samples) + '_K' + str(self.num_of_cls) + '_seed' + str(seed) + '.csv')
            label_path = os.path.join(self.root_dir, 'synthetic_data', self.data_kind + '#' + str(self.data_num)  + '_' + dr_str \
                + '_' + str(self.num_of_features) + 'x' + str(self.num_of_samples) + '_K' + str(self.num_of_cls) + '_seed' + str(seed) + '_label.csv')
	    print data_path
	    if os.path.exists(data_path): # the data file exists, just read it
                self.data_mat = self.read_data_from_csvfile(data_path)
                self.true_labels = self.read_data_from_csvfile(label_path)
                self.true_labels = self.true_labels[0, :] # since labels are stored as matrix, we just extrac row 0
                self.existed = True
	    else:
		if self.data_kind.startswith('syn'):
		    # we should generate synthetic dta with the linear model
		    self.data_mat, self.true_labels = self.gen_data_with_noise(self.num_of_features, self.num_of_samples, self.num_of_cls, \
			    self.data_num, self.has_outliers)
		    if self.dim_reduced:
		        self.data_mat = self.dim_reduction_by_spectral()
		        self.data_mat = self.data_mat.transpose()
		elif self.data_kind.startswith('2d'):
                    self.data_mat, self.true_labels = self.gen_2Data_with_3clusters(data_num = self.data_num)
		else:
		    raise ValueError('Error: no other synthetic datasets!')
                #print (self.root_dir)
                f_manager = FileManager(self.root_dir)
                f_manager.add_file(data_path)
                np.savetxt(data_path, np.asmatrix(self.data_mat), delimiter = ',')
                f_manager.add_file(label_path)
                np.savetxt(label_path, np.asmatrix(self.true_labels), delimiter = ',')
                self.existed = False

        print 'data_mat'
        print self.data_mat.shape

    '''
    The following functions presents different ways to obtain the data set
    '''

    def read_data_from_csvfile(self, f_path):
        """ This function is used to read data from a .csv file and
        return an array-like structure

        Args:
	   f_path (string): the absolute path of the .csv file

        Returns:
	   data_arr: data array (numpy array)

        Raises:
	   ValueError: if the file does not exist
        """
	if not os.path.exists(f_path):
            raise ValueError("Error: cannot find file: " + f_path)

        print (f_path)
        # read csv file into a pandas dataframe
        df = pd.read_csv(f_path, header = None)

        # convet the dataframe into a numpy mat
        data_arr = df.as_matrix()

        print (data_arr.shape)

        return data_arr

    def gen_data_without_noise(self, dim, num, rank):
	'''
	The function aims to generate the synthetic data by following
		X = W * H
	'''
	# set the range of elements in basis vector
        low_val = 0.0
        high_val = 1.0
	W = np.random.uniform(low_val, high_val, size = (dim, rank))
        H = np.zeros((rank, num))
	num_list = np.random.randint(np.maximum(1, 0.01 * num / rank), 1.4 * num / rank, rank - 1).tolist()
        num_list.append(num - np.sum(num_list))
        start_pos = 0
        for i in range(rank):
            H[i, start_pos:(start_pos + num_list[i])] = np.ones(num_list[i])
            start_pos = start_pos + num_list[i]
	X = W * np.asmatrix(H)
	true_labels = np.argmax(np.asarray(H), axis = 0)
	return X, true_labels, W

    def gen_data_with_noise(self, dim, num, rank, SNB, has_outliers):
        '''
        The function aims to generate the synthetic data following the model
                X = W * H + E
        where:
            columns of W are generated by uniform distribution in [0, 1]
            H is set to be [I, I, ..., I]  then performn row normalization
            E is drawn iid from standard guassian distribution
        so that X is obtained by W * H + E and then repeating the following step
            X = [X]_{+}
            E = X - W * H
            E = gamma * E
            X = W * H + E
        where is a scaling constant determined by the desired SNR, and {+} takes the nonnegative part
        of its argument. We may need to repeat the above steps several times, till we get a nonnegative
        H with desired SNR (SNR is defined by ||WH||_{F}^{2} / ||E||_{F}^{2})

        Detailed explanation can refer to the paper Learning From Hidden Traits: Joint Factor Analysis
        and Latent Clustering
        '''
        # set the range of elements in basis vector
        low_val = 0.0
        high_val = 1.0

        W = np.random.uniform(low_val, high_val, size = (dim, rank))
        H = np.zeros((rank, num))

        num_list = np.random.randint(np.maximum(1, 0.01 * num / rank), 1.4 * num / rank, rank - 1).tolist()
        num_list.append(num - np.sum(num_list))
        print 'num_list'
        print num_list
        start_pos = 0
        for i in range(rank):
            H[i, start_pos:(start_pos + num_list[i])] = np.ones(num_list[i])
            start_pos = start_pos + num_list[i]

        #np.random.shuffle(H.transpose())  # reorder the columns of H

        # construct the gaussian error for each cluster
        E = np.zeros((num, dim))
        cov_list = []
        mean = np.zeros(dim)
        start_row = 0
        for i in range(rank):
            #cov_list.append(np.random.uniform(0, 20, size = (dim, dim)).tolist())
            cov_list.append(np.eye(dim)) # set covariance matrix to delta * I
            dat = np.random.multivariate_normal(mean, cov_list[i], num_list[i])
            E[start_row:(start_row + num_list[i]), :] = np.asarray(dat) # copy data to E
            start_row = start_row + num_list[i] # update the start position for next dat
        E = E.transpose()
        #arr_idx = random.sample(range(num), num)
        arr_idx = np.random.choice(num, num, replace = False)
        #print 'random index'
        #print arr_idx
        # reorder the H and error
        H = H[:, arr_idx]
        E = E[:, arr_idx]

        H = np.asmatrix(H)
        #H = np.diag(np.diag(H * H.transpose()) ** (-0.5)) * H

        true_labels = np.argmax(np.asarray(H), axis = 0)

        #E = np.random.normal(size = (dim, num))
        print 'SNB: ' + str(SNB)
        ratio = 10 **(SNB / 10)
        while(True):
            X = W * H + E
            X = np.maximum(0, X)
            E = X - W * H
            SNR = LA.norm(W * H, 'fro') ** 2 / LA.norm(E, 'fro') ** 2
            print 'SNR :' + str(SNR)
            print 'ratio:' + str(ratio)
            print 'SNR - ratio : ' + str(abs(SNR - ratio))
            if abs(SNR - ratio) > 1e-10:
                gamma = np.sqrt(SNR / ratio)
                E = E * gamma
            else:
                break
        print '**SNB**: ' + str(LA.norm(W * H, 'fro') ** 2 / LA.norm(X - W * H, 'fro') ** 2)

        # add outliers to data mat
	if has_outliers:
	    num_outlier = int(0.05 * num)
            idx_list = np.random.randint(0, num, size = num_outlier)
            print 'where to insert outliers?'
            print idx_list
            X = np.asarray(X)
            # set the values of outliers
            # val = 1
            ol = np.random.uniform(size = (dim, num_outlier))
            for j in range(num_outlier):
                #X[:, idx_list[j]] = val * np.ones(dim)
                X[:, idx_list[j]] = ol[:, j] * 5


        #true_labels = np.argmax(np.asarray(H), axis = 0)
        #print true_labels
        if len(true_labels) != num:
            raise ValueError('Error: the length of data labels is not correct!')
        return np.asmatrix(X), true_labels


    def gen_2Data_with_3clusters(self, data_num):
	''' This function is to generate a set of 2-D data points with three clusters. And it
	is required that the centroids of at least two clusters are in a line originating from (0, 0)

	Args:
	   seed --float --- 
	   in_line (int) --- the numbef of clusters whose centroids are in a line originating from (0, 0)
	   balanced -- boolean -- True if the three clusters are of equal size, otherwise False   
	Returns:
	   a numpy array or matrix
	'''
	# randomly generate three points in the x-axis, 
	# if in_line = 2, the middle point will be considered as a centroid for a cluster, the rest two are centroids for two clusters in a line
	# if in_line = 3, the three points will be centroids for the three clusters in a line
	
	syn_data_mat = np.zeros(shape = (self.num_of_samples, self.num_of_features))
        labels = np.zeros(self.num_of_samples)
	if data_num == 1:  # incline = 2, balanced = True
	    in_line, balanced = 2, True
	    points = [5, 10, 15]	
	elif data_num == 2: # incline = 3, balanced = True
	    in_line, balanced = 2, True
	    points = [5, 20, 15]
	elif data_num == 3: # incline = 2, balanced = False
	    in_line, balanced = 2, True
	    points = [5, 30, 15]
	elif data_num == 4: # incline = 3, balanced = False
	    in_line, balanced = 2, True
	    points = [5, 30, 30]
	elif data_num == 5: # incline = 1, balanced = False
	    in_line, balanced = 1, False
	    points = [8, 10, 15]
	else:
            raise ValueError('Error: no other data types are supported!')

        if balanced:
            cls_sizes = np.ones(3, dtype = int) * int(self.num_of_samples / 3)
            cls_sizes[2] = self.num_of_samples - np.sum(cls_sizes[0:2])
        else:
            cls_sizes = np.random.randint(np.maximum(1, 0.1 * self.num_of_samples / self.num_of_cls),\
                 1.5 * self.num_of_samples / self.num_of_cls, self.num_of_cls)
            cls_sizes[2] = self.num_of_samples - np.sum(cls_sizes[0:2])
	    cls_sizes = np.array([200, 300, 500])
	
	print(cls_sizes)

	if in_line == 2:
	    start_row = 0
	    center = np.zeros(2)
	    cov = np.eye(2)
	    for i in range(3): 
		#point = np.random.random() * 10
		#print point
	        if i < 2:
		    center[0], center[1] = points[i], points[i] * 1.732
		else:	
		    center[0], center[1] = points[i], points[i] * 0.5
		dat = np.random.multivariate_normal(center, cov, int(cls_sizes[i]))
		while np.min(dat) < 0:
		    dat = np.random.multivariate_normal(center, cov, int(cls_sizes[i]))
		#print cls_sizes[i]
		syn_data_mat[start_row:(start_row + cls_sizes[i]), :] = dat
		labels[start_row:(start_row + cls_sizes[i])] = i
		start_row = start_row + cls_sizes[i]
	elif in_line == 3:
	    start_row = 0
	    center = np.zeros(2)
	    points = [5, 10, 15]
	    cov = np.eye(2)
	    for i in range(3):
		#center[:] = np.random.random() * 10
		center[0], center[1] = points[i], points[i]
                dat = np.random.multivariate_normal(center, cov, int(cls_sizes[i]))
                syn_data_mat[start_row:(start_row + cls_sizes[i]), :] = dat
		labels[start_row:(start_row + cls_sizes[i])] = i
                start_row = start_row + cls_sizes[i]
	elif in_line == 1:
	    start_row = 0
	    center = np.zeros(2)	
	    scale = [2.5, 1.5, 0.5]
	    #cov = np.eye(2) # can be changed
	    for i in range(3):
	        
		center[0], center[1] = points[i], points[i]  * scale[i]
		cov = np.diag(np.random.randn(1, 2).flatten())
		print cov
	   	dat = np.random.multivariate_normal(center, cov, int(cls_sizes[i]))
	    	while np.min(dat) < 0:
		    dat = np.random.multivariate_normal(center, cov, int(cls_sizes[i]))
                #print cls_sizes[i]
                syn_data_mat[start_row:(start_row + cls_sizes[i]), :] = dat
                labels[start_row:(start_row + cls_sizes[i])] = i
                start_row = start_row + cls_sizes[i]
	else:
	    raise ValueError('Error: no other number of clusters are in a line')
		
	if np.min(syn_data_mat) < 0:
	    print (np.min(syn_data_mat))
	    raise ValueError('The data mat has negative values!')

	return np.asmatrix(syn_data_mat).transpose(), labels
	

    def gen_inits_WH(self, init = 'random', seed = 1, H_ortho = True):
	''' The function is to initialize the factors W, H for nonnegative matrix factorization
        There are some options:
            1. random ------  generate W, H randomly
            2. kmeans ------  generate H based on cluster assignments obtained by Kmeans
                            then W = data_mat * H (since H is orthogonal)
            3. nmf    ------  use sklearn.nmf on data matrix firstly to get W, H for initialization
            4. kmeans++ ----  use heuristic strategy kmeans++ to get cluster assignment
                                    which can be used for H and W = data_mat * H

        Args:
            data (numpy array or mat): the input data
            init (string): the name of method used for generating the initializations
            rank (int): the rank for decomposition
            seed (float): the seed for random generator
        Returns:
            numpy matrix W and H
        '''
	ortho = 'ortho' if H_ortho else ''
	data_name = self.data_kind + str(self.data_num)

	initW_path = os.path.join(self.root_dir, 'inits', data_name, 'W' + str(seed) + '.csv')
        initH_path = os.path.join(self.root_dir, 'inits', data_name, 'H' + '_' + ortho + str(seed) + '.csv')
        if os.path.exists(initW_path) and os.path.exists(initH_path):
	    if seed < 100:
            	W_init = self.read_data_from_csvfile(initW_path)
            H_init = self.read_data_from_csvfile(initH_path)
        else:
	    (m, n) = self.data_mat.shape # get the size of data matrix to be decomposed
		
	    np.random.seed(seed)
            if init == 'random':
                abs_mat = np.absolute(self.data_mat)
                #print np.any(abs_mat < 0)
                avg = np.sqrt(abs_mat.mean() / self.num_of_cls)
                print 'mean: ' + str(abs_mat.mean())
                print 'rank: ' + str(self.num_of_cls)
                print 'avg: ' + str(avg)
                W_init = np.asmatrix(avg * np.random.random((m, self.num_of_cls)))
                H_init = np.asmatrix(avg * np.random.random((n, self.num_of_cls)))
            elif init == 'kmeans':
                km = sklearn_KMeans(n_clusters = self.num_of_cls).fit(self.data_mat.transpose())
                clusters = km.predict(self.data_mat.transpose())
                H_init = np.asmatrix(np.zeros((n, self.num_of_cls)))
                for i in range(len(clusters)):
                    H_init[i, clusters[i]] = 1
                H_init = H_init * np.diag(np.diag(H_init.transpose() * H_init) ** (-0.5))
                W_init = self.data_mat * H_init
            elif init == 'nmf':
                model = sklearn_NMF(n_components = self.num_of_cls, init = 'nndsvd', random_state = 0)
                W = model.fit_transform(self.data_mat.transpose())
                H = model.components_
                H_init = np.asmatrix(W)
                W_init = np.asmatrix(H).transpose()
            elif init == 'kmeans++':
                print 'using k++ initialization....'
                data_mat = self.data_mat.transpose()
                initial_centroids = np.ones((self.num_of_cls, m)) * (-1)
                ind_list = []
                idx = np.random.choice(n)
                ind_list.append(idx)
                initial_centroids[0, :] = data_mat[idx, :]
                while len(ind_list) < self.rank:
                    cent = initial_centroids[0:len(ind_list), :]
                    D2 = np.array([min([LA.norm(x - c) ** 2 for c in cent]) for x in data_mat])
                    probs = D2 / D2.sum()
                    cumprobs = probs.cumsum()
                    #r = random.random()
                    r = np.random.random()
                    idx = np.where(cumprobs >= r)[0][0]
                    ind_list.append(idx)
                    initial_centroids[len(ind_list) - 1, :] = data_mat[idx, :]
                print ind_list

                W_init = np.asmatrix(initial_centroids).transpose()
                distances = np.ones((m, self.num_of_cls)) * (-1)
                for centroid_idx in range(self.num_of_cls):
                    for data_idx in range(n):
                        distances[data_idx, centroid_idx] = LA.norm(data_mat[data_idx, :] - initial_centroids[centroid_idx, :])

                cluster_assignments = np.argmin(distances, axis = 1)
                temp_H = np.asmatrix(np.zeros((n, self.num_of_cls)))
                for j in range(n):
                    temp_H[j, cluster_assignments[j]] = 1

                #temp_H = np.diag(np.diag(temp_H * temp_H.transpose()) ** (-0.5)) * temp_H
                H_init = np.asmatrix(temp_H)
		
            else:
                raise ValueError('Error: invalid int parameter - init (None, random, kmeans, nmf)!!')

	    H_init = np.asmatrix(H_init.transpose())

	    if H_ortho:
		#H_init = np.asmatrix(H_init.transpose())
        	(ha, hb) = H_init.shape
        	ortho = LA.norm(H_init * H_init.transpose() - np.asmatrix(np.eye(ha)), 'fro')
        	print H_init * H_init.transpose()
        	if ortho > 1e-6:
            	    H = np.zeros((ha, hb))
                    ind = np.asarray(np.argmax(H_init, 0))[0, :]
                    for j in range(hb):
                        H[ind[j], j] = 1
                    H = np.asmatrix(H)
                    temp = np.diag(H * H.transpose())
                    if np.any(temp == 0):
                        print temp
                        raise ValueError("some rows of H are zeros!!!")
                    H = np.asmatrix(np.diag(temp ** (-0.5))) * H
                    H_init = H

	if seed >= 100:
            np.random.seed(seed)
	    (m, n) = self.data_mat.shape
	    
	    # find centers from the smallest clusters
	    cls_idx, cls_sizes = np.unique(self.true_labels, return_counts = True)
	    s_id = cls_idx[np.argmax(cls_sizes)]
	    id_list = np.where(self.true_labels == s_id)[0]
	    print s_id
	    print id_list

	    dis_mat = pdist(self.data_mat.transpose())
	    print np.argmin(dis_mat)
	    print np.unravel_index(dis_mat.argmin(), dis_mat.shape) 
	    print np.where(dis_mat==np.min(dis_mat[np.nonzero(dis_mat)]))
	    print 'select initial points -----'
	    select_idx = [997, 998, 999]
	    print select_idx
	    #print id_list
	    #select_idx = np.random.choice(id_list, self.num_of_cls, replace = False)
		
	    W_init = self.data_mat[:, select_idx]
	    #raise ValueError('TTEST!')
	    W_init = np.asmatrix(W_init)
	    print W_init.shape

	    # save generated initializations
            f_manager = FileManager(self.root_dir)
            f_manager.add_file(initW_path)
            np.savetxt(initW_path, np.asmatrix(W_init), delimiter = ',')
            f_manager.add_file(initH_path)
            np.savetxt(initH_path, np.asmatrix(H_init), delimiter = ',')

        return np.asmatrix(W_init), np.asmatrix(H_init)

    def visualize_data(self, dat_embeded = None, dim = 2, partition_idx = None, dat_path = None, \
        xlabel = '', ylabel = '', title = '', data_points = None):
        """ This function is used to visulize the data points in a 2-D space. Specifically, it
        will plot these data points in a figure after dimension-reduction with T-SNE. The
        data points belonging to the same partition will be plotted with the same color.

        Args:
            data_arr (numpy array): the data array to be plotted with each column being a data point
            dim (int): the dimension on which the data points to be plotted, default: 2
            partition_idx (list): a list of indexes indicating the partition of these data points
            dat_path (string): an absolute file path to store the 2D figure
            xlabel (string): a string to be shown on the x-axis of the 2D figure
            ylabel (string): a string to be shown on the y-axis of the 2D figure
            title (string): a string as the title of the figure
        Returns:
            None
        """
        if dim != 2:
            raise ValueError('Error: only 2-D figures are allowed now!')

        if dat_embeded is None: # generate it using tsne on data_arr
	    if self.data_mat.shape[0] > 2:
		print 'transform---------------'
                tsne = manifold.TSNE(n_components = dim, random_state = 0)
	        print 'data_mat:'+ str(self.data_mat.shape)
                dat_embeded = tsne.fit_transform(self.data_mat.transpose())
	    else:
		dat_embeded = np.asmatrix(np.copy(self.data_mat.transpose()))
	    if partition_idx is None:
		print 'partiton idx is none, use true labels!'
	    	partition_idx = self.true_labels
	    if dat_path is None:
	    	dat_folder = 'real_data' if self.is_real else 'synthetic_data'
            	dr_str = 'DR' if self.dim_reduced else ''
	    	dat_path = os.path.join(self.root_dir, dat_folder, self.data_kind, 'data' + dr_str + '#' + str(self.data_num) + '.pdf')
	    	title = self.data_kind + '#' + str(self.data_num)

        colormap = np.array(['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', \
                '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', \
                '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', \
                '#000000'])

	print 'data_embeded: ' + str(dat_embeded.shape)
	#map cluster indexs to 0-20
	pars = np.asarray(map(int, partition_idx))
	#print pars
	u_elms = np.unique(pars)
	ind = -1
	num = len(list(u_elms))
	if num > 20:
	    raise ValueError('The number of pars is greater than 20!!')
	step = int(20 / num)
	p_list = []
	for i in range(num):
	    if not u_elms[i] in p_list:
		ind = ind + step
		pars[pars == u_elms[i]] = ind
		p_list.append(u_elms[i])
	
	print pars
	if len(list(u_elms)) > self.num_of_cls or len(p_list) > self.num_of_cls:
	    print len(list(u_elms)), len(p_list), self.num_of_cls
	    raise ValueError('Error: the number of clusters are not consistent')

        color_used = colormap[list(pars)]
	dat_embeded = np.array(dat_embeded)
	print dat_embeded[0:5, 0:2]
	print dat_embeded.shape
        plt.figure()
        plt.scatter(dat_embeded[:, 0], dat_embeded[:, 1], c = color_used)
	if not data_points is None:
	    print data_points.shape
	    plt.scatter(data_points[:, 0], data_points[:, 1], c = 'blue', s=40)
        #plt.ylim(0, 20)
	#plt.xlim(0, 20)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig(dat_path, bbox_inches = 'tight')


    '''
    The following functions provide ways to share the data state
    '''

    def get_data_mat(self):
        return np.asmatrix(np.copy(self.data_mat))

    def get_data_name(self):
	return self.data_kind + '#' + str(self.data_num)

    def get_data_kind(self):
	return self.data_kind
	
    def get_data_num(self):
	return self.data_num

    def get_optimal_nmf_cost(self):
	if np.any(self.true_labels >= self.num_of_cls):
	    raise ValueError('Error: the label >= ' + str(self.num_of_cls))
	H_opt = np.zeros(shape = (self.num_of_cls, self.num_of_samples))
	for j in range(self.num_of_samples):  # for each data
	    cls_idx = self.true_labels[j]
	    H_opt[cls_idx, j] = 1
	# orthogonalize H_opt
	H_opt = np.asmatrix(H_opt)
        temp = np.diag(H * H.transpose())
        H = np.asmatrix(np.diag(temp ** (-0.5))) * H
	# get the optimal W
	W_opt = self.data_mat * H_opt.transpose()
	# compute the nmf-cost
        return LA.norm(self.data_mat - W_opt * H_opt, 'fro')**2 / LA.norm(self.data_mat, 'fro') ** 2	
	

    def get_dim(self):
        return self.num_of_features

    def get_num(self):
        return self.num_of_samples

    def is_labeled(self):
        return self.islabeled

    def get_labels(self):
        return self.true_labels

    def contain_negative_value(self):
        if np.nanmin(self.data_mat) < 0:
            return True
        else:
            return False

    def contain_zero_rows(self):
        if np.min(np.sum(self.data_mat, axis = 1)) == 0:
            return True
        else:
            return False

    def dim_reduction_by_spectral(self):
	'''
	The function performs dimension reduction using spectral analysis
	'''
	# create the affinity matrix
	print 'performing dimension reduction by spectral----------'
	print 'dimesion: ' + str(self.data_mat.shape)
	start_time = time.time()
	connectivity = kneighbors_graph(self.data_mat.transpose(), n_neighbors = 10, include_self = True)
        aff_mat = 0.5 * (connectivity + connectivity.T)
	affinity = aff_mat.todense()
	# create the graph laplacian matrix
	D = np.diag(np.asarray(np.sum(affinity, axis = 0))[0, :])
	lap_mat = D - affinity
	eig_vals, eig_vecs = LA.eig(lap_mat)
        sorted_eig_val_idx = np.argsort(eig_vals.real)
        sorted_eig_vecs = eig_vecs[:, sorted_eig_val_idx].real
        sorted_eig_vecs = sorted_eig_vecs[:, :self.num_of_cls]
	end_time = time.time()
	print 'time_used: ' + str(end_time - start_time)
	print 'dimension: ' + str(sorted_eig_vecs.shape)
	sc_data = np.asmatrix(sorted_eig_vecs).transpose()
	return sc_data

    '''
    The following functions gives several metric for clustering quality
    '''
    def get_cluster_assignments_from_matrix(self, indicator_mat):
        (da, db) = indicator_mat.shape
        print da, self.num_of_cls
        #if da != self.rank:
        #    raise ValueError('Error: the cluster indicator matrix has wrong dimension!')
        labels = np.argmax(np.asarray(indicator_mat), 0)
        if len(labels) != self.num_of_samples:
            raise ValueError('Error: the size of data samples must = the length of labels!')
        return labels

    def get_cluster_assignment_for_NMF(self, H):
        '''
        The function get the clustering quality of nmf by converting the cluster indicator matrix H
        to the corresponding cluster assignments which will be then compared with true labels
            1. specify the initial centroids by averging
            2. use kmeans for clustering
        '''
        (ha, hb) = H.shape
        print ha, hb
        cls_num = ha
        row_idx = []
        for i in range(ha):
            if np.max(H[i, :]) <= 0:
                row_idx.append(i)
                cls_num = cls_num - 1
        print 'the cluster number of H is ' + str(cls_num)
        H_reduced = np.delete(H, row_idx, 0)

        lt = H_reduced.argmax(0).tolist()[0] # get a list containing the maximum row index for each data point
        #print lt
        # get the set of unique cluster ids
        initial_cluster_ids = np.unique(lt)
        initial_cluster_indices = {}
        for cls_id in initial_cluster_ids:
            initial_cluster_indices[cls_id] = np.where(lt == cls_id)[0]
        #print initial_cluster_indices
        # for each initial cluster, compute the centroid by averaging all the data point inside
        #ha, hb = H_reduced.shape
        print ha, hb
        initial_centroids = np.zeros((cls_num, ha)) # each row corresponds to a centroid
        temp_H = H.transpose() # we transpose H to make each sample a row vector
        print temp_H.shape
        j = 0 # the ro index of initial_centroids
        for cls_id in initial_cluster_ids:
            num = len(initial_cluster_indices[cls_id])
            centroid = 0
            for i in range(num):
                centroid += temp_H[initial_cluster_indices[cls_id][i], :]
            centroid = centroid / num
            initial_centroids[j] = centroid # store the centroid
            j = j + 1
        print initial_centroids.shape
        print H.transpose().shape
        # kmeans clustering and return the cluster assiginemtns
        kmeans = KMeans(n_clusters = cls_num, init = initial_centroids).fit(H.transpose())
        cluster_assignments = kmeans.labels_
        return cluster_assignments


    def clustering_quality(self, cluster_assignments, true_classes):
        return self.calculate_purity(cluster_assignments, true_classes), \
	    self.calculate_rand_index(cluster_assignments, true_classes), \
            self.calculate_accuracy(cluster_assignments, true_classes), \
            self.calculate_NMI(cluster_assignments, true_classes)

    def calculate_purity(self, cluster_assignments, true_classes):
        '''
        Calculate the purity, a measurement of quality for the clustering results.
        Each cluster is assigned to the true class which is most frequent in the cluster.
        Using these classes, the percent accuracy is then calculated

        Input:
            cluster_assignments: an array contains cluster ids indicating the clustering
                                assignment of each data point with the same order in the data set

            true_classes: an array contains class ids indicating the true labels of each
                            data point with the same order in the data set

        Output:
            A number between 0 and 1. Poor clusterings have a purity close to 0 while a perfect clustering has a purity of 1
        '''
        # get the set of unique cluster ids
        cluster_ids = np.unique(cluster_assignments)
        cluster_indices = {}
        # find out the index of data points for each cluster in the data set
        for cls_id in cluster_ids:
            cluster_indices[cls_id] = np.where(cluster_assignments == cls_id)[0]
        # find out the index of data points for each class in the data set
        class_ids = np.unique(true_classes)
        class_indices = {}
        for cla_id in class_ids:
            class_indices[cla_id] = np.where(true_classes == cla_id)[0]

        # find out the true class which is most frequent for each cluster
        # record the number of correct classfications
        num_accuracy = 0
        for cls_id in cluster_ids:
            #max_id = class_ids[0]
            max_count = 0
            for cla_id in class_ids:
                tmp = len(np.intersect1d(cluster_indices[cls_id], class_indices[cla_id]))
                if max_count < tmp:
                    max_count = tmp
            num_accuracy = num_accuracy + max_count
        return float(num_accuracy) / len(cluster_assignments)

    def calculate_rand_index(self, cluster_assignments, true_classes):
        '''
        Calculate the Rand Index, a measurement of quality for the clustering results.
        It is essentially the percent accuracy of the clustering.

        The clustering is viewed as a series of decisions. There are N * (N-1) / 2
        pairs of samples in the dataset to be considered. The decision is considered
        correct if the pairs have the same label and are in the same cluster, or have
        different labels and are in different clusters. The number of correct decisions
        divided by the total number of decisions givens the Rand index

        Input:
            cluster_assignments: an array contains cluster ids indicating the clustering
                                assignment of each data point with the same order in the data set

            true_classes: an array contains class ids indicating the true labels of each
                        data point with the same order in the data set

        Output: A number between 0 and 1. Poor clusterings have a purity close to 0 while a perfect clustering has a purity of 1
        '''
        '''
        correct = 0
        total = 0
        # itertools.combinations will return all unordered pairs of indices of data points(0->N-1)
        for index_combo in itertools.combinations(range(len(true_classes)), 2):
            index1 = index_combo[0]
            index2 = index_combo[1]
            same_class = (true_classes[index1] == true_classes[index2])
            same_cluster = (cluster_assignments[index1] == cluster_assignments[index2])
            if same_class and same_cluster:
                correct = correct + 1
            elif not same_class and not same_cluster:
                correct = correct + 1
            else:
                pass # no nothing
            total = total + 1
        return float(correct) / total
        '''
        return adjusted_rand_score(cluster_assignments, true_classes)



    def calculate_accuracy(self, cluster_assignments, true_classes = None):
        '''
        The function calculate the clustering accurary which use the ratio of correctly
        clustered points over the total number of points (in [0, 1], the higher the better)

            AC = sum_{i from 1 to N}   delta(si, map(ri))   / N

        where N is the total number of documents and delta(x, y) is the delta function
        that equals to one if x = y and 0 otherwise. ri and si are the obtained cluster
        label and the true label for the i-th data sample. Map(ri) is the permutation
        mapping function that maps each cluster label ri to the equivalent label in true labels.

        Input:
            cluster_assignments: an array contains cluster ids indicating the clustering
                                assignment of each data point with the same order in the data set

            true_classes: an array contains class ids indicating the true labels of each
                            data point with the same order in the data set

        Output: A number between 0 and 1. Poor clusterings have a purity close to 0 while a perfect clustering has a purity of 1
        '''
	if true_classes is None:
	    true_classes = self.true_labels

        ca = self.best_map(true_classes, cluster_assignments)
	print 'best map'
	print ca
        return accuracy_score(ca, true_classes)

    def best_map(self, L1, L2):
        if L1.__len__() != L2.__len__():
            print('size(L1) must == size(L2)')

        Label1 = np.unique(L1)
        nClass1 = Label1.__len__()
        Label2 = np.unique(L2)
        nClass2 = Label2.__len__()

        nClass = max(nClass1, nClass2)
        G = np.zeros((nClass, nClass))
        for i in range(nClass1):
            for j in range(nClass2):
                G[i][j] = np.nonzero((L1 == Label1[i]) * (L2 == Label2[j]))[0].__len__()
        c = linear_assignment_.linear_assignment(-G.T)[:, 1]
        newL2 = np.zeros(L2.__len__())
        for i in range(nClass2):
            for j in np.nonzero(L2 == Label2[i])[0]:
                if len(Label1) > c[i]:
                    newL2[j] = Label1[c[i]]
        return newL2

    def calculate_NMI(self, cluster_assignments, true_classes):
        '''
        The function is to calculate NMI (the normalized mutual information) metric.

        Let C denote the set of clusters obtained from the ground truth and C' obtained
        from an algorithm. Their mutual information metric MI(C, C') is defined as follows:

        MI(C, C') = sum_{ci in C, cj' in C') p(ci, cj') * log2 (p(ci, cj') /(p(ci)p(cj')))

        where p(ci) and p(cj') are the probabilities that a data sample arbitrarily selected
        from the data set belongs to the clusters ci and cj', respectively, and p(ci, cj')
        is the joint probability that the arbitrarily selected data sample belongs to the
        clusters ci as well as cj' at the same time.

        Then the NMI is calculated as:

                NMI(C, C') = MI(C, C') / max(H(C), H(C'))

        where H(C) and H(C') are the entropies of C and C', respectively. It is easy to
        check that NMI(C, C') ranges from 0 to 1. NMI = 1 if two sets of clusters are identical,
        and NMI = 0 if the two sets are independent.

        Input:
            cluster_assignments: an array contains cluster ids indicating the clustering
                                assignment of each data point with the same order in the data set

            true_classes: an array contains class ids indicating the true labels of each
                        data point with the same order in the data set

        Output: A number between 0 and 1.
        '''
        return adjusted_mutual_info_score(cluster_assignments, true_classes)


    def get_corrected_error(self, cluster_assignments, cls_num):
        '''
        The function is to calculated the error ||X - WH||_{F}^{2} / ||X||_{F}^{2}
        with the optimal H obtained from the cluster assignments
        '''
        #print 'cluster_assignments1'
        #print cluster_assignments
        if len(cluster_assignments) != self.num_of_samples:
            raise ValueError('Error: the length of cluster index list must = num!')
        H = np.zeros((cls_num, self.num_of_samples))
        for j in range(self.num_of_samples):
            H[cluster_assignments[j], j] = 1
        H = np.asmatrix(H)
        H = np.asmatrix(np.diag(np.diag(H * H.transpose()) ** (-0.5))) * H
        W = self.data_mat * np.asmatrix(H).transpose()
        return LA.norm(self.data_mat - W * H, 'fro') ** 2 / LA.norm(self.data_mat, 'fro') ** 2

    def get_perfect_error(self):
        print 'true labels'
        print self.labels
        return self.get_corrected_error(self.labels)











