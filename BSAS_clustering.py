from matplotlib import style;  style.use('ggplot')
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
import BSAS_algorithm as bsalg
import BSAS_new as bsnew
#
#The data_norm_minimax contains the samples normalized with the minmax method
#The data_gaussian contains the samples normalized with mean = 0 and std = 1 (gaussian -normal- distribution)
#
data_norm_minimax = np.load('processed-data/user-feature-set-minimax.npy')
data_gaussian = np.load('processed-data/user-feature-set-stdscl.npy')

#model_new = bsnew.cluster_vectors(data_gaussian,0.28754986535759436, 0.8626495960727831, 0.01173672919826916,'final')

model_minmax = bsalg.BSAS()
model_minmax.fit_best(data_norm_minimax.T, first_time=False, n_times=50, dataname='minimax', plot_graph=True)

model_gaussian = bsalg.BSAS()
model_gaussian.fit_best(data_gaussian.T, first_time=False, n_times=50, dataname='stdscl', plot_graph=True)

theta_minmax, q_ = model_minmax.param()


theta_gaussian, q2_ = model_gaussian.param()

order_minimax = np.random.permutation(range(data_norm_minimax.shape[0]))
#order_minimax = np.load('processed-data/BSAS-data/order-minimax.npy')
#The order that gave the max. number of clusters
model_minmax.fit(data_norm_minimax.T, order_minimax)

order = np.random.permutation(range(data_gaussian.shape[0]))
#order = np.load('processed-data/BSAS-data/order-stdscl.npy')
# The order that gave the max. number of clusters
model_gaussian.fit(data_gaussian.T, order)

clusters, centroids = model_minmax.predict()
clusters, centroids


centroids_minimax = []
for key in centroids:
        centroids_minimax.append(centroids[key])

centroids_minimax = np.array(centroids_minimax)

clusters_minimax = []

for X in data_norm_minimax:
    tmp = cdist([X], centroids_minimax, 'euclidean')
    min_index, min_value = min(enumerate(tmp[0]), key=lambda p: p[1])
    clusters_minimax.append(min_index)

tmp = pd.DataFrame(data_norm_minimax)
tmp[19] = clusters_minimax

new_centroids_minimax = tmp.groupby([19]).mean()
new_centroids_minimax = new_centroids_minimax.values

print(clusters_minimax)
print(new_centroids_minimax)


clusters_, centroids_ = model_gaussian.predict()
clusters_, centroids_





centroids_stdscl = []
for key in centroids_:
    centroids_stdscl.append(centroids_[key])

centroids_stdscl = np.array(centroids_stdscl)

clusters_stdscl = []

for X in data_gaussian:
    tmp = cdist([X], centroids_stdscl, 'euclidean')
    min_index, min_value = min(enumerate(tmp[0]), key=lambda p: p[1])
    clusters_stdscl.append(min_index)

tmp = pd.DataFrame(data_gaussian)
tmp[19] = clusters_stdscl

new_centroids_stdscl = tmp.groupby([19]).mean()
new_centroids_stdscl = new_centroids_stdscl.values

print(clusters_stdscl)
print(new_centroids_stdscl)
np.save('processed-data/BSAS-data/order-minimax.npy', order_minimax)
np.save('processed-data/BSAS-data/clusters-minimax.npy', clusters_minimax)
np.save('processed-data/BSAS-data/centroids-minimax.npy', centroids_minimax)
np.save('processed-data/BSAS-data/order-stdscl.npy', order)
np.save('processed-data/BSAS-data/clusters-stdscl.npy', clusters_stdscl)
np.save('processed-data/BSAS-data/centroids-stdscl.npy', new_centroids_stdscl)