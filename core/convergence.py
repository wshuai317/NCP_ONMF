# This script defines the Convergence class which is used to
# to determine the convergence of an iterative optimization
# algorithm
#
# Author: Wang Shuai
# Time: 2018.02.23
##############################################################
from __future__ import division
from filemanager import FileManager
import numpy as np
import matplotlib.pyplot as plt
import os.path
from numpy import linalg as LA
from sklearn.cluster import KMeans


class Convergence(FileManager):

    def __init__(self, res_dir, TOL = 1e-4, max_iters = 500):
        self.TOL = TOL
        self.max_iters = max_iters
        self.res_dir = res_dir
        super(Convergence, self).__init__(self.res_dir)
        self.reset()

    def set_max_iters(self, num):
        self.max_iters = num

    def set_tolerance(self, tol):
        self.TOL = tol
	print 'set toll-----' + str(self.TOL) + ', ' + str(tol)

    def reset(self):
        ''' insert all the variables to be empty for storing later '''
        self.obj = []   # record objective value at each iteration
        self.prim_var = {}   # record primal variable values for each iteration
        self.prim_var_change = {}  # record the change of primal variable between two consective iterations
        self.dual_var = {}   # record dual variable values for each iteration
        self.dual_var_change = {} # record the change of dual variable between any two consective iterations
        self.fea_conditions = {} # record the satisfication of feasiblity conditions at each iteration

    ''' checking whether the stop criteria is satisfied or not '''
    def d(self):
	#print 'iter: ' + str(self.len()) + ', max_iter: ' + str(self.max_iters)
        if self.len() < 2:
            return False
        if self.len() > self.max_iters:
            print "hit max iters for convergence"
            return True
        # check for optimality gap of primal variables and dual variables
	'''
        # var containing the max difference of obj value and primal variables
        max_diff = abs(self.obj[len(self.obj) - 1] - self.obj[len(self.obj) - 2]) / abs(self.obj[len(self.obj) - 2])
        #max_diff = 0
        if len(self.prim_var) > 0: # if it has primal varaibles
            #prim_var_diff = 0
            for key in self.prim_var_change.keys():
                l = len(self.prim_var_change[key])
                if self.prim_var_change[key][l - 1] > max_diff:
                    max_diff = self.prim_var_change[key][l - 1]

        if len(self.dual_var) > 0: # if it has dual varialbes (some algorithm do not use dual variables)
            #dual_var_diff = 0
            for key in self.dual_var_change.keys():
                l = len(self.dual_var_change[key])
                if self.dual_var_change[key][l - 1] > max_diff:
                    max_diff = self.dual_var_change[key][l - 1]
        '''
	n_res = 0
	if len(self.prim_var) > 0: # if it has primal varaibles
            #prim_var_diff = 0
            for key in self.prim_var_change.keys():
                #l = len(self.prim_var_change[key])
                n_res += self.prim_var_change[key][-1]
        '''
	if len(self.dual_var) > 0: # if it has dual varialbes (some algorithm do not use dual variables)
            #dual_var_diff = 0
            for key in self.dual_var_change.keys():
                #l = len(self.dual_var_change[key])
                n_res += self.dual_var_change[key][-1]
        '''
       	#print 'the normalized residul of prim vars: ' + str(n_res) + ', TOL: ' + str(self.TOL)
        if n_res < self.TOL:
            print 'satisify the convergence criteria, prim_var_diff!!!'
            #print max_diff
            return True
        return False

    def len(self):
        return len(self.obj) # remove the initial value


    ''' insert the newly generated objective value into list'''
    def add_obj_value(self, obj):
        self.obj.append(obj)

    def add_prim_value(self, var_name, val):
        #print 'Before add prim value' + var_name
        #print val[0:4, 0:4]
        #print self.prim_var
        self.add_other_value(self.prim_var, var_name, val)
        #print self.prim_var[var_name]

        #print 'the number of elements in prim values:' + str(len(self.prim_var[var_name]))
        if len(self.prim_var[var_name]) > 1: # when adding new value of primal variables, we also compute and add the change of it if any
            #print 'len - 2'
            #print self.get_prim_val(var_name, len(self.prim_var[var_name]) - 2)[0:4, 0:4]
            #print 'len - 1'
            #print self.get_prim_val(var_name, len(self.prim_var[var_name]) - 1)[0:4, 0:4]
            prim_change = LA.norm(self.get_prim_val(var_name, len(self.prim_var[var_name]) - 1) \
                    - self.get_prim_val(var_name, len(self.prim_var[var_name]) - 2), 'fro') \
                    / LA.norm(self.get_prim_val(var_name, len(self.prim_var[var_name]) - 2), 'fro')
            #print LA.norm(self.get_prim_val(var_name, len(self.prim_var[var_name]) - 1) \
            #                             - self.get_prim_val(var_name, len(self.prim_var[var_name]) - 2), 'fro')
            #print 'prima change' + var_name + ' :' + str(prim_change) + ', index: ' + str(len(self.prim_var[var_name]) - 1) + str(len(self.prim_var[var_name]) - 2)
            #print 'primal change :' + str(prim_change)
            self.add_prim_change_value(var_name, prim_change)
            #print self.prim_var_change
        #print 'After add prim value'
        #print self.prim_var

    def add_prim_change_value(self, var_name, val):
        self.add_other_value(self.prim_var_change, var_name, val)

    def add_dual_value(self, var_name, val):
        self.add_other_value(self.dual_var, var_name, val)
        #print 'Called once'
        #print val
        if len(self.dual_var[var_name]) > 1: # when adding new value of dual variables, we also compute and add the change of it if any
            dual_change = LA.norm(self.get_dual_val(var_name, len(self.dual_var[var_name]) - 1) \
                    - self.get_dual_val(var_name, len(self.dual_var[var_name]) - 2), 'fro') \
                    / LA.norm(self.get_dual_val(var_name, len(self.dual_var[var_name]) - 2), 'fro')
            #print dual_change
            if dual_change < 0: raise ValueError(' the dual change value is negative !!!!')

            self.add_dual_change_value(var_name, dual_change)

    def add_dual_change_value(self, var_name, val):
        self.add_other_value(self.dual_var_change, var_name, val)

    def add_fea_condition_value(self, var_name, val):
        self.add_other_value(self.fea_conditions, var_name, val)

    def add_other_value(self, otherObj, var_name, val):
        if not var_name in otherObj.keys(): # if the var is not established in the dictionary
            otherObj[var_name] = []
        #print var_name
        #print val[0:5, 0:5]
        otherObj[var_name].append(np.copy(val))  # very important to use np.copy !!!!!

    ''' access the ojective value at a specific iteration or position'''
    def get_obj_val(self, pos):
        if pos < 0 or pos > len(self.obj) - 1:
            raise ValueError('Error: the index out of range when accessing obj values')
        return self.obj[pos]

    def get_prim_val(self, var_name, pos):
        return self.get_other_value(self.prim_var, var_name, pos)

    def get_dual_val(self, var_name, pos):
        return self.get_other_value(self.dual_var, var_name, pos)

    def get_last_obj_val(self):
        pos = len(self.obj) - 1
        return self.get_obj_val(pos)

    def get_last_prim_val(self, var_name):
        if not var_name in self.prim_var.keys():
            raise ValueError('Error:' + var_name + ' are not stored in the keys')
        pos = len(self.prim_var[var_name]) - 1
        return self.get_prim_val(var_name, pos)

    def get_last_last_prim_val(self, var_name):
        if not var_name in self.prim_var.keys():
            raise ValueError('Error:' + var_name + ' are not stored in the keys')
        l = len(self.prim_var[var_name])
        if l < 2: raise ValueError('Error: out of length')
        return self.get_prim_val(var_name, l - 2)

    def get_last_dual_val(self, var_name):
        pos = len(self.dual_var[var_name]) - 1
        return self.get_dual_val(var_name, pos)

    def get_last_prim_change(self, var_name):
        pos = len(self.prim_var_change[var_name]) - 1
        return self.get_prim_change_value(var_name, pos)
    def get_last_dual_change(self, var_name):
        pos = len(self.dual_var_change[var_name]) - 1
        return self.get_dual_change_value(var_name, pos)

    def get_last_fea_condition_value(self, var_name):
        pos = len(self.fea_conditions[var_name]) - 1
        return self.get_fea_condition_value(var_name, pos)

    def get_prim_change_value(self, var_change_name, pos):
        return self.get_other_value(self.prim_var_change, var_change_name, pos)

    def get_dual_change_value(self, var_change_name, pos):
        return self.get_other_value(self.dual_var_change, var_change_name, pos)

    def get_fea_condition_value(self, fea_name, pos):
        return self.get_other_value(self.fea_conditions, fea_name, pos)

    def get_other_value(self, otherObj, var_name, pos):
        if not var_name in otherObj.keys():
            raise ValueError('Error:' + var_name + ' are not stored in the keys')
        if pos < 0 or pos > len(otherObj[var_name]) - 1:
            raise ValueError('Error: the index out of range when accessing ' + var_name)
        return otherObj[var_name][pos]

    ''' store the objecitve value to a file at a specific path'''
    def store_obj_val(self, dirname):
        self.add_dir(dirname)
        ph = os.path.join(dirname, 'objlist.csv')
        #print 'store obj path:' + path

        np.savetxt(ph, np.asarray(self.obj), delimiter = ",")

    def store_prim_val(self, pos, dirname): # since each primal variable value is a matrix, we just store the element at a position
        #pathlist = path.split('.')
        #if len(pathlist) != 2: # if the path string does not contain the suffix for file type or contains more than one '.'
        #    raise ValueError('Error: the input path for storing primal values is not valid')
        for var_name in self.prim_var.keys():
            real_path = os.path.join(dirname, var_name + '.csv')
            self.add_file(real_path)
            np.savetxt(real_path, np.asmatrix(self.prim_var[var_name][pos]), delimiter = ",")
            if var_name == 'H': # if it is the coeffient matrix, I also generate the cluste indicator matrix from it
                h_data = np.asmatrix(self.prim_var[var_name][pos])
                (ha, hb) = h_data.shape
                #print h_data[0, :]
                h_indicator = np.zeros_like(h_data)
                for j in range(hb):
                    h_indicator[np.argmax(h_data[:, j]), j] = 1
                norm_h_indicator = np.diag(np.diag(h_indicator * h_indicator.transpose()) ** (-0.5)) * h_indicator
                #print h_indicator[0, :]
                real_path = os.path.join(dirname, var_name + '_max.csv')
                self.add_file(real_path)
                np.savetxt(real_path, np.asmatrix(norm_h_indicator), delimiter = ',')

                # use kmeans to obtain the indicator matrix again
                # kmeans clustering and return the cluster assiginemtns
                H = h_data
                (ha, hb) = H.shape
                print ha, hb
                cls_num = ha
                row_idx = []
                for i in range(ha):
                    if np.max(H[i, :]) <= 0:
                        row_idx.append(i)
                        cls_num = cls_num - 1
                #print 'the cluster number of H is ' + str(cls_num)
                H_reduced = np.delete(H, row_idx, 0)

                lt = H_reduced.argmax(0).tolist()[0] # get a list containing the maximum row index for each data point
                #print lt
                # get the set of unique cluster ids
                initial_cluster_ids = np.unique(lt)
                initial_cluster_indices = {}
                for cls_id in initial_cluster_ids:
                    initial_cluster_indices[cls_id] = np.where(lt == cls_id)[0]

                initial_centroids = np.zeros((cls_num, ha)) # each row corresponds to a centroid
                temp_H = H.transpose() # we transpose H to make each sample a row vector
                #print temp_H.shape
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

                kmeans = KMeans(n_clusters = cls_num, init = initial_centroids).fit(h_data.transpose())
                cluster_assignments = kmeans.labels_

                h_kmeans = np.zeros_like(h_data)
                for j in range(len(cluster_assignments)):
                    h_kmeans[cluster_assignments[j], j] = 1
                norm_h_kmeans = np.diag(np.diag(h_kmeans * h_kmeans.transpose()) ** (-0.5)) * h_kmeans
                real_path = os.path.join(dirname, var_name + '_kmeans.csv')
                self.add_file(real_path)
                np.savetxt(real_path, np.asmatrix(norm_h_kmeans), delimiter = ',')



    '''plot the figure to show the convergence of obj value'''
    def plot_convergence_obj(self, dirname):
        ylabel = '||X - WH||_{F}^2'
        filename = 'obj.png'
        self.plot_var_change(self.obj, dirname, filename, ylabel)

    def plot_convergence_prim_var(self, dirname):
        ylabel = '||W(k) - W(k-1)||/||W(k-1)|| + ||H(k) - H(k-1)||/||H(k-1)||'
        filename = 'prim_var_change.png'
        #print len(self.prim_var_change.keys())
        var_change = np.asarray(self.prim_var_change[self.prim_var_change.keys()[0]]) # choose the first key-value for temp
        for key in self.prim_var_change.keys():
            if key != self.prim_var_change.keys()[0]:
                var_change = var_change + np.asarray(self.prim_var_change[key])
        ph = os.path.join(dirname, 'prim_change.csv')
        np.savetxt(ph, np.asarray(var_change), delimiter = ",")
        self.plot_var_change(var_change.tolist(), dirname, filename, ylabel)

    def plot_convergence_dual_var(self, dirname):
        ylabel = '||Lamba(k) - Lamba(k-1)||/||Lamba(k-1)||'
        filename = 'dual_var_change.png'
        var_change = np.asarray(self.dual_var_change[self.dual_var_change.keys()[0]]) # choose the first key-value for temp
        #print var_change.tolist()
        for key in self.dual_var_change.keys():
            if key != self.dual_var_change.keys()[0]:
                var_change = var_change + np.asarray(self.dual_var_change[key])
                if np.min(var_change) <= 0:
                    raise ValueError(' The var change of dual varialbes is negative !!!!!!')
        #print var_change.tolist()
        self.plot_var_change(var_change.tolist(), dirname, filename, ylabel)

    def plot_convergence_fea_condition(self, dirname):
        ''' different from the change of primal and dual variables, for each feasibility condition, we plot a figure '''
        for key in self.fea_conditions.keys():
            ylabel = key
            filename = key + '.png'
            # store the fea conditions value at first
            ph = os.path.join(dirname, key + '.csv')
            np.savetxt(ph, np.asarray(self.fea_conditions[key]), delimiter = ",")
            self.plot_var_change(self.fea_conditions[key], dirname, filename, ylabel)


    def plot_var_change(self, var_change, dirname, filename, ylabel):
        ''' for the convergence of primal variables, dual variables, or feasiblity, we plot the prim_var_change and the sum of change of all the primal variables'''
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        #print filename
        #print var_change
        ax.semilogy(var_change)
        ax.set_xlabel('Iters: k')
        ax.set_ylabel(ylabel)
        if self.is_valid_request(dirname):
            #if the dirname is abosulte path
            self.add_dir(dirname)
            path = os.path.join(dirname, filename)
        else:
            # if the dirname is not abosulte path, then put it under the root dir
            self.add_dir(os.path.join(self.fileroot, dirname))
            path = os.path.join(self.fileroot, dirname, filename)
        plt.savefig(path)

    def save_data(self, data, dirname, filename):
        if self.is_valid_request(dirname):
            #if the dirname is abosulte path
            self.add_dir(dirname)
            path = os.path.join(dirname, filename)
        else:
            # if the dirname is not abosulte path, then put it under the root dir
            self.add_dir(os.path.join(self.fileroot, dirname))
            path = os.path.join(self.fileroot, dirname, filename)

        np.savetxt(path, np.asarray(data), delimiter = ",")









