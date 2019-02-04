import PyQt5
import matplotlib.pyplot as plt
from matplotlib import style;  style.use('ggplot')
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist
import pandas as pd
import BSAS_algorithm as bsalg

#
#The X_minimax contains the samples normalized with the minmax method
#The X_stdscl contains the samples normalized with mean = 0 and std = 1 (gaussian -normal- distribution)
#
X_minimax = np.load('processed-data/user-feature-set-minimax.npy')
X_stdscl = np.load('processed-data/user-feature-set-stdscl.npy')


clf = bsalg.BSAS()
clf.fit_best(X_minimax.T, first_time=False, n_times=50, dataname='minimax', plot_graph=True)

clf2 = bsalg.BSAS()
clf2.fit_best(X_stdscl.T, first_time=False, n_times=50, dataname='stdscl', plot_graph=True)

theta_, q_ = clf.specs()
theta_, q_

theta2_, q2_ = clf2.specs()
theta2_, q2_

order_minimax = np.random.permutation(range(X_minimax.shape[0]))
# order_minimax = np.load('comp-data/2-bsas-comp-data/order-minimax.npy')
#The order that gave the max. number of clusters
clf.fit(X_minimax.T, order_minimax)

order = np.random.permutation(range(X_stdscl.shape[0]))
# order = np.load('comp-data/2-bsas-comp-data/order-stdscl.npy')
# The order that gave the max. number of clusters
clf2.fit(X_stdscl.T, order)

clusters, centroids = clf.predict()
clusters, centroids


centroids_minimax = []
for key in centroids:
        centroids_minimax.append(centroids[key])

centroids_minimax = np.array(centroids_minimax)

clusters_minimax = []

for X in X_minimax:
    tmp = cdist([X], centroids_minimax, 'euclidean')
    min_index, min_value = min(enumerate(tmp[0]), key=lambda p: p[1])
    clusters_minimax.append(min_index)

tmp = pd.DataFrame(X_minimax)
tmp[19] = clusters_minimax

new_centroids_minimax = tmp.groupby([19]).mean()
new_centroids_minimax = new_centroids_minimax.values

print(clusters_minimax)
print(new_centroids_minimax)


clusters_, centroids_ = clf2.predict()
clusters_, centroids_





centroids_stdscl = []
for key in centroids_:
    centroids_stdscl.append(centroids_[key])

centroids_stdscl = np.array(centroids_stdscl)

clusters_stdscl = []

for X in X_stdscl:
    tmp = cdist([X], centroids_stdscl, 'euclidean')
    min_index, min_value = min(enumerate(tmp[0]), key=lambda p: p[1])
    clusters_stdscl.append(min_index)

tmp = pd.DataFrame(X_stdscl)
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