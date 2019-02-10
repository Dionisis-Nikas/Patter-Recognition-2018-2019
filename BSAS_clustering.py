#######################################################################
# CLUSTERING USING THE BSAS ALGORITHM AND OUR GAUSSIAN SCALED DATASET #
#######################################################################



#######################################################################
#                   IMPORT LIBRARIES                                  #
#######################################################################
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
import BSAS_algorithm as bsalg

#######################################################################
#                      LOAD THE DATASET                               #
#######################################################################
#The data_gaussian contains the samples normalized with mean = 0 and std = 1 (gaussian -normal- distribution)
#
data_gaussian = np.load('processed-data/user-feature-set-gaussian.npy')



#######################################################################
#      FIT THE DATA TO THE BSAS ALGORITHM AND GET THE ESTIMATIONS     #
#######################################################################
model_gaussian = bsalg.BSAS()
model_gaussian.fit_best(data_gaussian.T,n_times=5, load_precalculated=False)

##################################################################################
#   FROM THE ESTIMATIONS GET THE BEST ORDER THAT GAVE THE MAX NUMBER OF CLUSTERS #
#               AND COMPUTE CLUSTERS AND THEIR CENTROIDS                         #
##################################################################################
order = np.load('/Users/Dennis/Downloads/Movielens-data-classification-master/comp-data/2-bsas-comp-data/order-gaussian.npy')

# The order that gave the max. number of clusters
model_gaussian.fit(data_gaussian.T, order)

clusters_, centroids_ = model_gaussian.predict()

centroids_stdscl = []
for key in centroids_:
    centroids_stdscl.append(centroids_[key])

centroids_stdscl = np.array(centroids_stdscl)

clusters_stdscl = []

for sample in data_gaussian:
    tmp = cdist([sample], centroids_stdscl, 'euclidean')
    mip = enumerate(tmp[0])
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