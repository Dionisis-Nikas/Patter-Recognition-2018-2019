#######################################################################
# CLUSTERING USING THE BSAS ALGORITHM AND OUR GAUSSIAN SCALED DATASET #
#######################################################################



#######################################################################
#                   IMPORT LIBRARIES                                  #
#######################################################################
import matplotlib.pyplot as plt
from matplotlib import style; style.use('ggplot')
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from tqdm import tqdm

#######################################################################
#                    LOAD THE STANDARD SCLAED DATASET                 #
#######################################################################

#TO ΑΡΧΕΙΟ ΑΠΟ ΤΙΣ ΠΡΩΤΕΣ ΕΠΕΞΕΡΓΑΣΙΕΣ. ΒΡΙΣΚΕΤΕ ΣΤΟ ΤΕΛΟΣ ΚΑΙ ΕΙΝΑΙ ΑΥΤΟ ΠΟΥ ΤΟΥ ΕΧΕΙ ΔΩΘΕΙ ΤΟ ΟΝΟΜΑ x_stdscl
gausian_data = np.load('processed-data/user-feature-set-gaussian.npy')

############################################################################
#CREATE THE LINKAGE MATRIX AND PLOT THE HIERARCHICAL CLUSTERING DENDOGRAM  #
############################################################################

matrix_g = linkage(gausian_data, 'complete')
plt.figure(1, figsize=(25, 10)) #DENDOGRAM FIGURE SIZE
plt.title('Hierarchical Clustering Dendrogram (Full Version)')
plt.xlabel('Gausian_data[i]')
plt.ylabel('Distance')
dendrogram(
    matrix_g,
    leaf_rotation=90,                                                                            # ΣΤΡΕΦΕΙ ΤΙΣ ΕΤΙΚΕΤΕΣ ΤΟΥ Χ ΚΑΤΑ 90 ΜΟΙΡΕΣ ΓΙΑ ΝΑ ΣΧΗΜΑΤΙΣΕΙ ΤΙΣ ΓΩΝΙΕΣ
    leaf_font_size=8                                                                            #ΤΟ ΜΕΓΕΘΟΣ ΤΗΣ ΓΡΕΑΜΜΗΣ
)
plt.show()




######################################################################################
# ALSO PLOT THE SHORT VERSION OF THE DENDOGRAM FOR BETTER UNDERSTANDING AND ANALYSIS #
######################################################################################

plt.figure(2, figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram (Short Version)')
plt.xlabel('Sample [i]')
plt.ylabel('Distance')
dendrogram(
    matrix_g,
    truncate_mode='lastp',       # show only the last p merged clusters
    p=20,                        # show only the last p merged clusters
    show_leaf_counts=False,      # otherwise numbers in brackets are counts
    leaf_rotation=90,
    leaf_font_size=12,
    show_contracted=True,        # to get a distribution impression in truncated branches
)
plt.show()




#######################################################################
#                  CLUSTER DETERMINATION USING ELBOW METHOD           #
#######################################################################

#THE DENDOGRAM GAVE US 2-9 CLUSTERS SO WE WILL RUN THE K-MEANS (number of clusters) --> FOR ALL THE VALUES 2-9

distortion_vector = []
clusters_vector = [1, 2, 3, 4, 5, 6, 7, 8, 9]
for k in tqdm(clusters_vector):
    kmean_alg = KMeans(n_clusters=k, n_init=20, precompute_distances=True, random_state=0, verbose=2)
    kmean_alg.fit(gausian_data)
    kmean_alg.fit(gausian_data)
    distortion_vector.append(sum(np.min(cdist(gausian_data, kmean_alg.cluster_centers_, 'euclidean'), axis=1)) / gausian_data.shape[0])

#######################################################################
#                      PLOT THE ELBOW METHOD                          #
#######################################################################

plt.figure(2, figsize=(25, 10))
plt.plot(clusters_vector, distortion_vector, 'bx-')
plt.xlabel('Clusters')
plt.ylabel('Distortion')
plt.title('Elbow Method')
plt.show()



#######################################################################
#     USE DATA FROM ELBOW METHOD AND TRIM DENDOGRAM                   #
#######################################################################

max_d = 6.10 #TRIM DENDOGRAM DISTANCE TO 6.10 SINCE CLUSTERS IS 4
hierarchical_clusters = fcluster(matrix_g, max_d, criterion='distance') #DISTANCE RELATED CLUSTERING

#GROUP THE DATASET BASED ON EACH CLUSTER THEN CALCULATE THE MEAN OF EACH ONE AND SAVE IT AS THE CENTROIDS
gausian_data_hierarchical = pd.DataFrame(gausian_data)
gausian_data_hierarchical[19] = hierarchical_clusters
hierarchical_centroids = gausian_data_hierarchical.groupby([19]).mean()


#######################################################################
#                  SAVE DATA TO NPY ARRAY FILES                       #
#######################################################################
np.save('processed-data/hierarchical_clusters.npy', hierarchical_clusters)
np.save('processed-data/hierarchical_centroids.npy', hierarchical_centroids)