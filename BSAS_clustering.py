
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
import BSAS_algorithm as bsalg



#The data_gaussian contains the samples normalized with mean = 0 and std = 1 (gaussian -normal- distribution)
#
data_gaussian = np.load('processed-data/user-feature-set-gaussian.npy')



model_gaussian = bsalg.BSAS()
model_gaussian.fit_best(data_gaussian.T, load_precalculated=False, n_times=50)
theta_gaussian, q_ = model_gaussian.param()

order2 = np.random.permutation(range(data_gaussian.shape[0]))
order = np.load('processed-data/BSAS-data/order-gaussian.npy')
# The order that gave the max. number of clusters
model_gaussian.fit(data_gaussian.T, order)

clusters_, centroids_ = model_gaussian.predict()

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


#np.save('processed-data/BSAS-data/order-gaussian.npy', order)
np.save('processed-data/BSAS-data/clusters-gaussian.npy', clusters_stdscl)
np.save('processed-data/BSAS-data/centroids-gaussian.npy', new_centroids_stdscl)