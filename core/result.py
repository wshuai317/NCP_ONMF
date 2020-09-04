#############################################################################################
# This script is to define a Result class for NMF which contains all the result info generated
# during the iterative solving process including the obj value, the change of X, Y. It also
# should contains the methods for analyzing the result.
#
# @author Wang Shuai
# @date 2018.02.28
##############################################################################################

from filemanager import FileManager
import numpy as np
import matplotlib.pyplot as plt
import os.path

class Result(FileManager):

    ''' initialize all the variables to be empty for storing later'''
    def __init__(self, outputdir):
        super(Result, self).__init__(outputdir)
        self.reset()

    def reset(self):
        self.obj = []
        self.X = []
        self.Y = []

    '''insert the newly generated objective value into list'''
    def add_obj_val(self, obj):
        self.obj.append(obj)

    def add_X_val(self, x):
        self.X.append(x)

    def add_Y_val(self, y):
        self.Y.append(y)

    '''access the ojective value at a specific position'''
    def get_obj_val(self, pos):
        if pos < 0 or pos > len(self.obj) - 1:
            print "Error: the index out of range"
            return None
        return self.obj[pos]

    def get_X_val(self, pos):
        if pos < 0 or pos > len(self.X) - 1:
            print "Error: the index out of range"
            return None
        return self.X[pos]


    def get_Y_val(self, pos):
        if pos < 0 or pos > len(self.Y) - 1:
            print "Error: the index out of range"
            return None
        return self.Y[pos]

    ''' store the objective value to a file at a specific path'''
    def store_obj_val(self, path):
        self.add_file(path)
        np.savetxt(path, np.asarray(self.obj), delimiter = ",")

    def store_X_val(self, path):
        self.add_file(path)
        np.savetxt(path, np.asarray(self.X), delimiter = ",")

    def store_Y_val(self, path):
        self.add_file(path)
        np.savetxt(path, np.asarray(self.Y), delimiter = ",")

    '''plot the figure to show the convergence of objective value of a given nmf object instance'''
    def plot_convergence_obj(self, nmfObj, dirname):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(nmfObj)
        ax.set_xlabel('Iters: k')
        ax.set_ylabel('||X - WH||_{F}^2')
        if self.is_valid_request(dirname):
            #if the dirname is abosulte path
            self.add_dir(dirname)
            path = os.path.join(dirname, 'obj.jpg')
        else:
            # if the dirname is not abolute path, then put it under the root dir
            self.add_dir(os.path.join(self.fileroot, dirname))
            path = os.path.join(self.fileroot, dirname, 'obj.jpg')
        plt.savefig(path)


    def plot_convergence_var(self, varChange, dirname):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(varChange)
        ax.set_xlabel('Iters: k')
        ax.set_ylabel('||W(k) - W(k-1)||/||W(k-1)|| + ||H(k) - H(k-1)||/||H(k-1)||')
        if self.is_valid_request(dirname):
            #if the dirname is abosulte path
            self.add_dir(dirname)
            path = os.path.join(dirname, 'var.jpg')
        else:
            # if the dirname is not abosulte path, then put it under the root dir
            self.add_dir(os.path.join(self.fileroot, dirname))
            path = os.path.join(self.fileroot, dirname, 'var.jpg')
        plt.savefig(path)

    ''' Here I use a dict to store the list data for each feasibility condition, the key is the condition'''
    ''' Sample data: {'||H - P||' : data1, '||W - P||': data2}'''
    def plot_convergence_feas(self, dirname, dictData):
        fig, axes = plt.subplots(1, len(dictData))
        i = 0
        for key, value in dictData.items():
            axes[i].plot(value)
            axes[i].set_xlabel('Iters: k')
            print key
            axes[i].set_ylabel(key)
            i = i + 1

        if self.is_valid_request(dirname):
            #if the dirname is abosulte path
            self.add_dir(dirname)
            path = os.path.join(dirname, 'feas.jpg')
        else:
            # if the dirname is not absolute path, then put it under the root dir
            self.add_dir(os.path.join(self.fileroot, dirname))
            path = os.path.join(self.fileroot, dirname, 'feas.jpg')
        plt.savefig(path)


if __name__ == "__main__":
    res = Result("/home/wshuai/Work/test_dir");
    arr_x = np.array([6, 7, 8, 9])
    res.add_obj_val(90)
    res.add_obj_val(3.4)
    res.add_obj_val(7.8)
    dat = [1, 2, 3, 4, 5]
    dt = {}
    dt['|H-P|'] = dat
    dt['Y - P'] = dat
    dt['X - Y'] = dat
    res.plot_convergence_feas("res", dt)
    res.plot_convergence_obj(arr_x, "res")
    res.store_obj_val("/home/wshuai/Work/test_dir/1.csv")
