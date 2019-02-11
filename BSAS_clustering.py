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
data_gaussian = np.load('processed-data/gaussian-dataset.npy')



#######################################################################
#      FIT THE DATA TO THE BSAS ALGORITHM AND GET THE ESTIMATIONS     #
#######################################################################
model_gaussian = bsalg.BSAS_model()
model_gaussian.calc_BSAS_data(data_gaussian.T,n_times=5, load_precalculated=False)

##################################################################################
#   FROM THE ESTIMATIONS GET THE BEST ORDER THAT GAVE THE MAX NUMBER OF CLUSTERS #
#               AND COMPUTE CLUSTERS AND THEIR CENTROIDS                         #
##################################################################################
order = np.random.permutation(range(data_gaussian.shape[0]))

# The order that gave the max. number of clusters
model_gaussian.run_BSAS(data_gaussian.T, order)

clusters, centroids = model_gaussian.predict()

centroids_gaussian = []
for key in centroids:
    centroids_gaussian.append(centroids[key])
centroids_gaussian = np.array(centroids_gaussian)

clusters_gaussian = []

for sample in data_gaussian:
    tmp = cdist([sample], centroids_gaussian, 'euclidean')
    count = enumerate(tmp[0])
    index, minimum_value = min(count, key=lambda p: p[1])
    clusters_gaussian.append(index)

final_data = pd.DataFrame(data_gaussian)
final_data[19] = clusters_gaussian

final_centroids_gaussian = final_data.groupby([19]).mean()
final_centroids_gaussian = final_centroids_gaussian.values

print(clusters_gaussian)
print(final_centroids_gaussian)


np.save('processed-data/BSAS-data/clusters-gaussian.npy', clusters_gaussian)
np.save('processed-data/BSAS-data/centroids-gaussian.npy', final_centroids_gaussian)
