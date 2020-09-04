from __future__ import division
import numpy as np
from scipy.cluster.hierarchy import linkage, cophenet, dendrogram
#import matplotlib
import matplotlib.pyplot as plt
import os.path as path
import pandas as pd



def connectivity(coef_mat = None):
    '''
    Compute the connectivity matrix for the smaples based on their mixture coefficients.

    The connectivity matrix C is a symmetric matrix which shows the shared memebership of the samples:
        entry C_ij is 1 iff sample i and sample j belong to the same cluster, 0 otherwise.
    Sample assignment is determined by its largest value index.

    Input:
        coef_mat ------ the coefficient matrix returned by NMF-like methods
    Output:
        connectivity matrix
    '''
    if coef_mat is None:
        raise ValueError('Error: input is missing!')
    (ca, cb) = coef_mat.shape
    conn_mat = np.zeros((cb, cb))
    assignments = np.argmax(coef_mat, axis = 0)
    print assignments
    for i in range(cb):
        for j in range(cb):
            if assignments[i] == assignments[j]:
                conn_mat[i, j] = 1
    #print conn_mat[0:10, 0:10]
    return np.asmatrix(conn_mat)


def consensus(dict_mats = None, num = 0):
    '''
    Compute consensus matrix as the mean connectivity matrix across multiple runs of the factorization.
    It has been proposed by Brunet2004 to help visualize and measure the stability of the clusters obtained by NMF.

    Input:
        dict_mats ------ a dictionary of coef matrices obtained by NMF-like methods with key = run#, value = matrix, such as {'1': A, '2': B, ...}
        num       ------ sample number
    Output:
        consensus matrix
    '''
    if dict_mats is None:
        raise ValueError('Error: input is missing!')

    cons_mat = np.asmatrix(np.zeros((num, num)))
    for _, val in dict_mats.items():
        cons_mat += connectivity(val)
    cons_mat = cons_mat / len(dict_mats)

    return cons_mat

def coph_cor(cons_mat = None):
    '''
    Compute cophenetic correlation coefficient of consensus matrix, generally obtained from multiple NMF runs.

    The cophetic correlation coefficient is measure which indicates the dispersion of the consensus matrix which
    is the average of connectivity matrices. It measures the stability of the clusters obtained by NMF.

    It is computed as the Pearson correlation of two distance matrices:
        1. The first one is the distance between samples induced by the consensus matrix
        2. The second one is the distance between samples induced by the linkage used in the reordering of
            the consensus matrix

    In a perfect consensus matrix, cophenetic correlation equals to 1. When the entries in consensus matrix are
    scattered between 0 and 1, the cophenetic correlation is < 1.
    '''
    if cons_mat is None:
        raise ValueError('Error: input is missing!')
    # upper off-diagonal elements of consensus
    avec = np.array([cons_mat[i, j] for i in range(cons_mat.shape[0] - 1) for j in range(i + 1, cons_mat.shape[1])])

    # consensus entries are similarities, conversion to distances
    Y = 1 - avec
    #Y[abs(Y) < 1e-20] = 0.0
    #numpy.clip(Y, 0, 1, Y)
    #print any(Y < 0)
    #print np.min(Y)
    Z = linkage(Y, method = 'average')
    #print np.min(Z)
    #print any(Z < 0)
    #Z[abs(Z) < 1e-20] = 0.0
    #cophenetic correlation coefficient of a hierarchical clustering defined by the linkage matrix Z
    # and matrix Y from which Z was generated
    return cophenet(Z, Y)[0], Z

def heatmap(data, ax = None, cbar_kw = {}, cbar_label = "", **kwargs):
    '''
    Create a heatmap from a numpy array

    Input:
        data    ------  A 2D numpy array of shape(M, N)
        ax      ------  A matplotlib.axes.Axes instance to which the heatmap is plotted.
                        If not provided, use current axes or create a new one.
        cbar_kw ------  A dictionary with arguments to :meth: 'matplotlib.Figure.colorbar'.
        cbarlabel-----  The label for the colorbar
    All other arguments are directly passed on to the imshow call
    '''
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # create the colorbar
    cbar = ax.figure.colorbar(im, ax = ax, **cbar_kw)
    cbar.ax.set_ylabel(cbar_label, rotation = -90, va = 'bottom')

    return im, cbar

def consensus_map(cons_mat = None):

    cop, Z  = coph_cor(cons_mat)

    print 'cophetic correlation coefficient'
    print cop

    dn = dendrogram(Z, no_plot = True)

    # get the labels in the dendrogram true to reorder samples in consensus matrix
    ind = dn['ivl']
    ind = map(int, ind)
    #print ind
    #print len(ind)
    #print 'consensus map'
    #print cons_mat
    cons_mat = np.asarray(cons_mat)
    new_mat = (cons_mat[:, ind])[ind, :]
    #print new_mat
    #print new_mat.shape
    #print cons_mat.shape
    fig, ax = plt.subplots()


    heatmap(np.asarray(new_mat), ax = ax, cmap = "bwr", cbar_label = '')
    plt.title('SNCP, SNR = -5, Cophenetic correlation coefficient: ' + str(cop))
    plt.show()

if __name__ == "__main__":
    coef_mats = {}
    res_dir = '/home/wl318/cloud/JMLR2020/matlab/'
    for i in range(1, 21):
        #data_path = path.join(res_dir, 'onmf', 'sncp2_0_2', 'epsilon3e-03&rho1.1', 'rank10', 'data-3', 'seed' + str(i), 'H.csv')
        data_path = path.join(res_dir, 'sncp_vs_nsncp', 'syn_otlr#-5', 'sncp', 'H' + str(i) + '.csv')
        df = pd.io.parsers.read_csv(data_path, header = None)
        coef_mats[i] = df.as_matrix()
        #print coef_mats[i][0, 0:5]

    cons_mat = consensus(coef_mats, 1000)
    consensus_map(cons_mat)








