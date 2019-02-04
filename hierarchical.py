import PyQt5
import matplotlib.pyplot as plt
from matplotlib import style; style.use('ggplot')
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster





#TO ΑΡΧΕΙΟ ΑΠΟ ΤΙΣ ΠΡΩΤΕΣ ΕΠΕΞΕΡΓΑΣΙΕΣ. ΒΡΙΣΚΕΤΕ ΣΤΟ ΤΕΛΟΣ ΚΑΙ ΕΙΝΑΙ ΑΥΤΟ ΠΟΥ ΤΟΥ ΕΧΕΙ ΔΩΘΕΙ ΤΟ ΟΝΟΜΑ x_stdscl
X = np.load('comp-data/1-preprocessing-comp-data/user-feature-set-stdscl.npy')

#ΔΗΜΙΟΥΡΓΙΑ ΤΟΥ ΤΟΥ ΠΙΝΑΚΑ
ZC = linkage(X, 'complete')

#ΥΠΟΛΟΓΙΣΜΟΣ ΔΕΝΔΡΟΓΡΑΜΜΑΤΟΣ
plt.figure(1, figsize=(25, 10)) #ΤΟ ΜΕΓΕΘΟΣ ΤΟΥ ΔΕΝΔΡΟΓΡΑΜΜΑΤΟΣ
plt.title('Hierarchical Clustering Dendrogram -- Complete-Linkage') #ΤΙΤΛΟΣ
plt.xlabel('X[i]') #ΑΞΟΝΑΣ Χ
plt.ylabel('distance') #ΑΞΟΝΑΣ Υ
dendrogram(
    ZC,
    leaf_rotation=90,  # ΣΤΡΕΦΕΙ ΤΙΣ ΕΤΙΚΕΤΕΣ ΤΟΥ Χ ΚΑΤΑ 90 ΜΟΙΡΕΣ ΓΙΑ ΝΑ ΣΧΗΜΑΤΙΣΕΙ ΤΙΣ ΓΩΝΙΕΣ
    leaf_font_size=8,  #ΤΟ ΜΕΓΕΘΟΣ ΤΗΣ ΓΡΕΑΜΜΗΣ
)
plt.show()

distortions = []
K = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for i in K:
    kmeanTest = KMeans(n_clusters=k, n_init=20, n_jobs=-1, precompute_distances=True, random_state=0, verbose=2)
    kmeanTest.fit(X);
    kmeanTest.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanTest.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# ΔΗΜΙΟΥΡΓΙΑ ΤΟΥ PLOT ΓΙΑ ΤΟ elbow
plt.figure(2, figsize=(25, 10))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

#ΑΠΟ ΤΗΝ elbow ΜΕΘΟΔΟ ΟΙ ΟΜΑΔΟΠΟΙΗΣΕΙΣ ΠΟΥ ΕΓΙΝΑΝ ΕΙΝΑΙ 4 ΟΠΩΣ ΚΑΙ ΤΟ ΠΡΟΤΕΙΝΟΜΕΝΟ ΤΟΥ bsas (4)


max_d = 6.10 #ΜΕΓΙΣΤΗ ΑΠΟΣΤΑΣΗ ΑΠΟ ΤΟ ΚΕΝΤΡΟ
clusters_ = fcluster(ZC, max_d, criterion='distance') #ΔΗΜΙΟΥΡΓΙΑ ΟΜΑΔΩΝ ΜΕ ΚΡΙΤΗΡΙΟ ΤΗΝ ΑΠΟΣΤΑΣΗ
clusters_


tmp = pd.DataFrame(X) #????
tmp[19] = clusters_ #????

centroids_ = tmp.groupby([19]).mean() #ΔΗΜΙΟΥΡΓΙΑ ΚΕΝΤΡΟΕΙΔΩΝ
centroids_

np.save('comp-data/3b-hierarchical-clustering-comp-data/clusters_.npy', clusters_) #ΣΩΖΟΥΜΕ ΤΑ ΑΠΟΤΕΛΕΜΣΑΤΑ ΣΕ ΑΡΧΕΙΑ
np.save('comp-data/3b-hierarchical-clustering-comp-data/centroids_.npy', centroids_)#ΣΩΖΟΥΜΕ ΤΑ ΑΠΟΤΕΛΕΜΣΑΤΑ ΣΕ ΑΡΧΕΙΑ