import numpy as np
import scipy.stats as stats
import matplotlib
import ot
import scipy as sp
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.cluster import KMeans

#%% parameters and data generation
n_source = 30 # nb samples
n_target = 40 # nb samples

mu_source = np.array([[0, 0], [0, 5]])
cov_source = np.array([[1, 0], [0, 1]])

mu_target = np.array([[10, 1], [10, 5], [10, 10]])
cov_target = np.array([[1, 0], [0, 1]])

num_clust_source = mu_source.shape[0]
num_clust_target = mu_target.shape[0]

xs = np.vstack([ot.datasets.get_2D_samples_gauss(n_source/num_clust_source,  mu_source[i,:], cov_source) for i in range(num_clust_source)])
xt = np.vstack([ot.datasets.get_2D_samples_gauss(n_target/num_clust_target,  mu_target[i,:], cov_target) for i in range(num_clust_target)])

ind_clust_source = np.vstack([i*np.ones(n_source/num_clust_source) for i in range(num_clust_source)])
ind_clust_target = np.vstack([i*np.ones(n_target/num_clust_target) for i in range(num_clust_target)])

#%% individual OT

# uniform distribution on samples
n1 = len(xs)
n2 = len(xt)
a, b = np.ones((n1,)) / n1, np.ones((n2,)) / n2 

# loss matrix
M = ot.dist(xs, xt)
M /= M.max()

#OT
#G_ind = ot.emd(a, b, M)
lambd = 1e-3
G_ind = ot.sinkhorn(a, b, M, lambd)

## plot OT transformation for samples
#ot.plot.plot2D_samples_mat(xs, xt, G_ind, c=[.5, .5, 1])
#pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
#pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
#pl.legend(loc=0)
#pl.title('OT matrix with samples')

#%% clustered OT

#find CM of clusters
kmeans = KMeans(n_clusters=num_clust_source, random_state=0).fit(xs)
CM_source = kmeans.cluster_centers_
kmeans = KMeans(n_clusters=num_clust_target, random_state=0).fit(xt)
CM_target = kmeans.cluster_centers_

# loss matrix
M_clust = ot.dist(CM_source, CM_target)
M_clust /= M_clust.max()

# uniform distribution on CMs
n1_clust = len(CM_source)
n2_clust = len(CM_target)
a_clust, b_clust = np.ones((n1_clust,)) / n1_clust, np.ones((n2_clust,)) / n2_clust # uniform distribution on samples

#OT
G_clust = ot.emd(a_clust, b_clust, M_clust)

## plot OT transformation for CMs
#ot.plot.plot2D_samples_mat(CM_source, CM_target, G_clust, c=[.5, .5, 1])
#pl.plot(CM_source[:, 0], CM_source[:, 1], 'ok', markersize=10,fillstyle='full')
#pl.plot(CM_target[:, 0], CM_target[:, 1], 'ok', markersize=10,fillstyle='full')
#pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
#pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
#pl.legend(loc=0)
#pl.title('OT matrix with samples, CM')

#%% OT figures
f, axs = pl.subplots(1,2,figsize=(10,4))

sub=pl.subplot(121)
ot.plot.plot2D_samples_mat(xs, xt, G_ind, c=[.5, .5, 1])
pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
pl.legend(loc=0)
sub.set_title('OT by samples')

sub=pl.subplot(122)
ot.plot.plot2D_samples_mat(CM_source, CM_target, G_clust, c=[.5, .5, 1])
pl.plot(CM_source[:, 0], CM_source[:, 1], 'ok', markersize=10,fillstyle='full')
pl.plot(CM_target[:, 0], CM_target[:, 1], 'ok', markersize=10,fillstyle='full')
pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
pl.legend(loc=0)
sub.set_title('OT by CMs')

#%%
##########################################
#%% consistency measures
##########################################

#%% parameters and data generation
n_source = 5 # nb samples
n_target = 5 # nb samples

mu_source = np.array([[0, 0], [0, 5]])
cov_source = np.array([[1, 0], [0, 1]])

#mu_target = np.array([[10, 1], [10, 5], [10, 10]])
mu_target = np.array([[10, 3], [10, 5]])
cov_target = np.array([[1, 0], [0, 1]])

num_clust_source = mu_source.shape[0]
num_clust_target = mu_target.shape[0]

ind_clust_source = np.hstack([i*np.ones(n_source/num_clust_source) for i in range(num_clust_source)])
ind_clust_target = np.hstack([i*np.ones(n_target/num_clust_target) for i in range(num_clust_target)])

iter_all=25

G_small_ind = np.zeros((iter_all,num_clust_source,num_clust_target)) 
G_small_ind_row = np.zeros((iter_all,num_clust_source*num_clust_target)) 
G_small_clust_row = np.zeros((iter_all,num_clust_source*num_clust_target)) 

for j in range(iter_all):

    xs = np.vstack([ot.datasets.get_2D_samples_gauss(n_source/num_clust_source,  mu_source[i,:], cov_source) for i in range(num_clust_source)])
    xt = np.vstack([ot.datasets.get_2D_samples_gauss(n_target/num_clust_target,  mu_target[i,:], cov_target) for i in range(num_clust_target)])
    n1 = len(xs)
    n2 = len(xt)
    a, b = np.ones((n1,)) / n1, np.ones((n2,)) / n2 
    M = ot.dist(xs, xt)
    M /= M.max()
    lambd = 1e-3
    G_ind_iter = ot.sinkhorn(a, b, M/10, lambd)

    temp_rows = np.vstack([np.sum(G_ind_iter[np.where(ind_clust_source==i)[0],:],axis=0) for i in range(num_clust_source)])
    temp_col = np.vstack([np.sum(temp_rows[:,np.where(ind_clust_target==i)[0]],axis=1) for i in range(num_clust_target)]).T
#    G_small_ind[j,:,:] = np.divide(temp_col,np.tile(np.sum(temp_col,axis=1),(num_clust_target,1)).T)
    G_small_ind_row[j,:] = np.hstack(np.divide(temp_col,np.tile(np.sum(temp_col,axis=1),(num_clust_target,1)).T))

    kmeans = KMeans(n_clusters=num_clust_source, random_state=0).fit(xs)
    CM_source = kmeans.cluster_centers_
    labels_source = kmeans.labels_
    kmeans = KMeans(n_clusters=num_clust_target, random_state=0).fit(xt)
    CM_target = kmeans.cluster_centers_
    labels_target = kmeans.labels_
    M_clust = ot.dist(CM_source, CM_target)
    M_clust /= M_clust.max()
    n1_clust = len(CM_source)
    n2_clust = len(CM_target)
    a_clust, b_clust = np.ones((n1_clust,)) / n1_clust, np.ones((n2_clust,)) / n2_clust # uniform distribution on samples
    G_clust = ot.sinkhorn(a_clust, b_clust, M_clust/10,lambd)
    G_clust_samples_rows = np.zeros((G_ind_iter.shape[0],num_clust_target))
    for i in range(num_clust_source):
        G_clust_samples_rows[np.where(labels_source==i)[0],:] = G_clust[i,:]
    G_clust_samples_col = np.zeros((G_ind_iter.shape[0],G_ind_iter.shape[1]))
    for i in range(num_clust_target):
        G_clust_samples_col[:,np.where(labels_target==i)[0]] = np.tile(G_clust_samples_rows[:,i],(len(np.where(labels_target==i)[0]),1)).T
    temp_rows = np.vstack([np.sum(G_clust_samples_col[np.where(ind_clust_source==i)[0],:],axis=0) for i in range(num_clust_source)])
    temp_col = np.vstack([np.sum(temp_rows[:,np.where(ind_clust_target==i)[0]],axis=1) for i in range(num_clust_target)]).T
#    G_small_ind[j,:,:] = np.divide(temp_col,np.tile(np.sum(temp_col,axis=1),(num_clust_target,1)).T)
    G_small_clust_row[j,:] = np.hstack(np.divide(temp_col,np.tile(np.sum(temp_col,axis=1),(num_clust_target,1)).T))
    
    
#    np.vstack([np.tile(G_ind_iter[np.where(labels_source==i)[0],:],axis=0) for i in range(num_clust_source)])

cor_mat_ind = np.corrcoef(G_small_ind_row)
iu1_ind = np.triu_indices(cor_mat_ind.shape[0],1)
cor_ind=np.nanmean(np.abs(cor_mat_ind[iu1_ind]))

cor_mat_clust = np.corrcoef(G_small_clust_row)
iu1_clust = np.triu_indices(cor_mat_clust.shape[0],1)
cor_clust=np.nanmean(np.abs(cor_mat_clust[iu1_clust]))

print("individual correlation = "+str(cor_ind) + ", clustered correlation = "+str(cor_clust))
