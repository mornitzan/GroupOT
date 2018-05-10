
from sklearn import manifold, datasets, decomposition
import matplotlib
from scipy.interpolate import griddata

import sklearn
from scipy.spatial.distance import cdist

import ot

import scipy as sp
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D 

import networkx as nx
from sklearn.neighbors import kneighbors_graph

import random

from sklearn.cluster import KMeans
from __future__ import division
from numpy import linalg as LA
from sklearn.neighbors import NearestNeighbors

#%% synthetic biological data (for batch effect in single cell sequencing data)

# parameters and data generation
ns = 1000 # number of samples in first experiment
nt = 1000 # number of samples in second experiment
num_dim = 100 # number of dimensions of single cell data
proj_mat = np.random.normal(3,1,(2,num_dim)) # projection matrix from 2d to num_dim d
rand_vec1 = np.random.normal(2,1,(num_dim,1)) # bias vector for first experiment
rand_vec2 = np.random.normal(28,1,(num_dim,1)) # bias vector for second experiment

# parameters for mixture of gaussians representing low-d mixture of cell types
mu_s1 = np.array([0, 0])
cov_s1 = np.array([[1, 0], [0, 1]])
mu_s2 = np.array([5, 9])
cov_s2 = np.array([[1, 0], [0, 1]])
mu_s3 = np.array([11, 7])
cov_s3 = np.array([[1, 0], [0, 1]])

# sampling single cells for first experiment
xs1 = ot.datasets.get_2D_samples_gauss(int(ns*0.2), mu_s1, cov_s1)
xs2 = ot.datasets.get_2D_samples_gauss(int(ns*0.3), mu_s2, cov_s1)
xs3 = ot.datasets.get_2D_samples_gauss(int(ns*0.5), mu_s3, cov_s1)
xs = np.vstack((xs1,xs2,xs3))
labs = np.hstack((np.ones(int(ns*0.2)),2*np.ones(int(ns*0.3)),3*np.ones(int(ns*0.5)))) # cell type labels
datas = np.dot(xs,proj_mat) + rand_vec1.T # single cell data 

# sampling single cells for second experiment
xt1 = ot.datasets.get_2D_samples_gauss(int(nt*0.05), mu_s1, cov_s1)
xt2 = ot.datasets.get_2D_samples_gauss(int(nt*0.65), mu_s2, cov_s1)
xt3 = ot.datasets.get_2D_samples_gauss(int(nt*0.3), mu_s3, cov_s1)
xt = np.vstack((xt1,xt2,xt3))
labt = np.hstack((np.ones(int(nt*0.05)),2*np.ones(int(nt*0.65)),3*np.ones(int(nt*0.3)))) # cell type labels
datat = np.dot(xt,proj_mat) + rand_vec2.T # single cell data 

n1 = datas.shape[0]
n2 = datat.shape[0]
a, b = np.ones((n1,)) / n1, np.ones((n2,)) / n2 # uniform distribution on samples

# distance matrix
M = ot.dist(datas, datat)
M /= M.max()

##embed data for visualization
emb_dat = decomposition.PCA(n_components=2).fit_transform(np.vstack((datas,datat)))

#% plot embedded samples
#emb_dat = manifold.TSNE(n_components=2).fit_transform(np.vstack((datas,datat)))
pl.plot(emb_dat[0:ns, 0], emb_dat[0:ns, 1], '+b', label='Source samples')
pl.plot(emb_dat[ns:, 0], emb_dat[ns:, 1], 'xr', label='Target samples')
pl.legend(loc=0)
pl.title('Source and target distributions')

#OT
G0 = ot.emd(a, b, M)
#lambd = 1e-3
#G0 = ot.sinkhorn(a, b, M, lambd)

##% plot OT from source to target
#pl.imshow(G0, interpolation='nearest')
colors = ['red','green','blue']
ot.plot.plot2D_samples_mat(emb_dat[0:ns, :], emb_dat[ns:2*ns, :], G0, c=[.5, .5, 1])
pl.scatter(emb_dat[0:ns, 0], emb_dat[0:ns, 1], marker = 'o', label='Source samples', c=labs, cmap=matplotlib.colors.ListedColormap(colors))
pl.scatter(emb_dat[ns:, 0], emb_dat[ns:, 1], marker = '^', label='Target samples', c=labt, cmap=matplotlib.colors.ListedColormap(colors))
pl.legend()
pl.title('OT matrix with samples')


mat_results = np.zeros((3,2))
for i in range(1,4):
    mat_results[i-1,0] = (1/len(np.where(labs==i)[0]))*((np.sum(G0[np.where(labs==i)[0][:, None],np.where(labt==i)[0]]))/(np.sum(G0[np.where(labs==i)[0][:, None],:])))/(len(np.where(labt==i)[0])/len(labt))
    mat_results[i-1,1] = (1/len(np.where(labs==i)[0]))*((np.sum(G0[np.where(labs==i)[0][:, None],np.where(labt!=i)[0]]))/(np.sum(G0[np.where(labs==i)[0][:, None],:])))/(len(np.where(labt!=i)[0])/len(labt))

print(mat_results)

# mutual nearest neighbors
all_data = np.vstack((datas,datat))
all_data = np.divide(all_data,np.tile(LA.norm(all_data,axis=1),(num_dim,1)).T)

temp_dist = cdist(all_data,all_data, 'euclidean')
temp_dist[0:ns-1,0:ns-1] = np.nan
temp_dist[(ns-1):len(temp_dist),(ns-1):len(temp_dist)] = np.nan
temp_ord = np.argsort(temp_dist,axis=1)

knn = 5
temp_dist1 = np.zeros((len(temp_dist),len(temp_dist)))
for i in range(len(temp_dist)):
    temp_dist1[i,temp_ord[i,0:knn]]=1

G_mnn = np.multiply(temp_dist1,temp_dist1.T)[0:ns,(ns):len(temp_dist)]    

mat_results_mnn = np.zeros((3,2))
for i in range(1,4):
    mat_results_mnn[i-1,0] = (1/len(np.where(labs==i)[0]))*((np.sum(G0[np.where(labs==i)[0][:, None],np.where(labt==i)[0]]))/(np.sum(G_mnn[np.where(labs==i)[0][:, None],:])))/(len(np.where(labt==i)[0])/len(labt))
    mat_results_mnn[i-1,1] = (1/len(np.where(labs==i)[0]))*((np.sum(G0[np.where(labs==i)[0][:, None],np.where(labt!=i)[0]]))/(np.sum(G_mnn[np.where(labs==i)[0][:, None],:])))/(len(np.where(labt!=i)[0])/len(labt))

print(mat_results_mnn)

# correlation + majority vote
temp_dist_mm = cdist(all_data,all_data, 'correlation')
temp_dist_mm[0:ns-1,0:ns-1] = np.nan
temp_dist_mm[(ns-1):len(temp_dist),(ns-1):len(temp_dist)] = np.nan

G_mm = temp_dist_mm[0:ns,(ns):len(temp_dist)]

mat_results_mm = np.zeros((3,2))
for i in range(1,4):
    mat_results_mm[i-1,0] = (1/len(np.where(labs==i)[0]))*((np.sum(G_mm[np.where(labs==i)[0][:, None],np.where(labt==i)[0]]))/(np.sum(G_mm[np.where(labs==i)[0][:, None],:])))/(len(np.where(labt==i)[0])/len(labt))
    mat_results_mm[i-1,1] = (1/len(np.where(labs==i)[0]))*((np.sum(G_mm[np.where(labs==i)[0][:, None],np.where(labt!=i)[0]]))/(np.sum(G_mm[np.where(labs==i)[0][:, None],:])))/(len(np.where(labt!=i)[0])/len(labt))

print(mat_results_mm)

