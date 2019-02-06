import matplotlib.pyplot as plt
from matplotlib import style; style.use('ggplot')
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from tqdm import tqdm




#TO ΑΡΧΕΙΟ ΑΠΟ ΤΙΣ ΠΡΩΤΕΣ ΕΠΕΞΕΡΓΑΣΙΕΣ. ΒΡΙΣΚΕΤΕ ΣΤΟ ΤΕΛΟΣ ΚΑΙ ΕΙΝΑΙ ΑΥΤΟ ΠΟΥ ΤΟΥ ΕΧΕΙ ΔΩΘΕΙ ΤΟ ΟΝΟΜΑ x_stdscl
gausian_data = np.load('processed-data/user-feature-set-stdscl.npy')

#ΔΗΜΙΟΥΡΓΙΑ ΤΟΥ ΤΟΥ ΠΙΝΑΚΑ
linkage_matrix = linkage(gausian_data, 'complete')

#ΥΠΟΛΟΓΙΣΜΟΣ ΔΕΝΔΡΟΓΡΑΜΜΑΤΟΣ
plt.figure(1, figsize=(25, 10)) #ΤΟ ΜΕΓΕΘΟΣ ΤΟΥ ΔΕΝΔΡΟΓΡΑΜΜΑΤΟΣ
plt.title('Hierarchical Clustering Dendrogram -- Complete-Linkage') #ΤΙΤΛΟΣ
plt.xlabel('gausian_data[i]') #ΑΞΟΝΑΣ Χ
plt.ylabel('distance') #ΑΞΟΝΑΣ Υ
dendrogram(
    linkage_matrix,
    leaf_rotation=90,  # ΣΤΡΕΦΕΙ ΤΙΣ ΕΤΙΚΕΤΕΣ ΤΟΥ Χ ΚΑΤΑ 90 ΜΟΙΡΕΣ ΓΙΑ ΝΑ ΣΧΗΜΑΤΙΣΕΙ ΤΙΣ ΓΩΝΙΕΣ
    leaf_font_size=8,  #ΤΟ ΜΕΓΕΘΟΣ ΤΗΣ ΓΡΕΑΜΜΗΣ
)
plt.show()

plt.figure(2, figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram -- Complete-Linkage (truncated)')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    linkage_matrix,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=20,  # show only the last p merged clusters
    show_leaf_counts=False,  # otherwise numbers in brackets are counts
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.show()

distortions = []
K = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for k in tqdm(K):
    kmeanTest = KMeans(n_clusters=k, n_init=20, precompute_distances=True, random_state=0, verbose=2)
    kmeanTest.fit(gausian_data)
    kmeanTest.fit(gausian_data)
    distortions.append(sum(np.min(cdist(gausian_data, kmeanTest.cluster_centers_, 'euclidean'), axis=1)) / gausian_data.shape[0])

# ΔΗΜΙΟΥΡΓΙΑ ΤΟΥ PLOT ΓΙΑ ΤΟ elbow
plt.figure(2, figsize=(25, 10))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

#ΑΠΟ ΤΗΝ elbow ΜΕΘΟΔΟ ΟΙ ΟΜΑΔΟΠΟΙΗΣΕΙΣ ΠΟΥ ΕΓΙΝΑΝ ΕΙΝΑΙ 4 ΟΠΩΣ ΚΑΙ ΤΟ ΠΡΟΤΕΙΝΟΜΕΝΟ ΤΟΥ bsas (4)


max_d = 6.10 #ΜΕΓΙΣΤΗ ΑΠΟΣΤΑΣΗ ΑΠΟ ΤΟ ΚΕΝΤΡΟ
clusters_ = fcluster(linkage_matrix, max_d, criterion='distance') #ΔΗΜΙΟΥΡΓΙΑ ΟΜΑΔΩΝ ΜΕ ΚΡΙΤΗΡΙΟ ΤΗΝ ΑΠΟΣΤΑΣΗ
clusters_


tmp = pd.DataFrame(gausian_data) #????
tmp[19] = clusters_ #????

centroids_ = tmp.groupby([19]).mean() #ΔΗΜΙΟΥΡΓΙΑ ΚΕΝΤΡΟΕΙΔΩΝ
centroids_

np.save('processed-data/clusters_.npy', clusters_) #ΣΩΖΟΥΜΕ ΤΑ ΑΠΟΤΕΛΕΜΣΑΤΑ ΣΕ ΑΡΧΕΙΑ
np.save('processed-data/centroids_.npy', centroids_)#ΣΩΖΟΥΜΕ ΤΑ ΑΠΟΤΕΛΕΜΣΑΤΑ ΣΕ ΑΡΧΕΙΑ