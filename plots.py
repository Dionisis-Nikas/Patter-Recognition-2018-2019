#######################################################################
#      PLOTTING THE DATASET AND PERFOMANCE CLUSTEING ANALYSIS         #
#######################################################################



#######################################################################
#                   IMPORT LIBRARIES                                  #
#######################################################################
import matplotlib.pyplot as plt
from matplotlib import style;  style.use('ggplot')

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as pca_decom


#######################################################################
#      PLOTTING THE DATASET AND PERFOMANCE CLUSTERING ANALYSIS         #
#######################################################################



def plot_pca_cluster(type='None',label_data=None):
    tmp = pd.DataFrame(X_pca)
    tmp[2] = label_data
    colors = ['y','r','b','k','c','m','g','w']
    pca = tmp.groupby([2]).mean()
    pca = pca.values
    title = 'PCA RESULT PLOT WITH %s ALGORITHM'%(type)
    plt.figure(1, figsize=(25, 8))
    plt.title(title)  # Ο ΤΙΤΛΟΣ
    plt.xlabel('X')  # ΑΞΟΝΑΣ Χ
    plt.ylabel('Y')  # ΑΞΟΝΑΣ Υ
    i=0
    for cluster in (range(max(label_data)+1)):
        label = 'Class %s'%(i+1)
        label_centroid = 'Class %s centroid'%(i+1)
        color = 'C%s'%(i)
        # ΤΟ scatter ΧΡΗΣΙΜΟΠΟΙΕΤΑΙ ΓΙΑ ΝΑ ΣΚΟΡΠΙΣΕΙ ΤΑ ΣΤΟΙΧΕΙΑ ΚΑΙ ΝΑ ΤΑ ΒΑΛΕΙ ΣΕ ΕΝΑ ΓΡΑΦΗΜΑ
        if (np.array_equal(label_data,label_hiercl))and(cluster==0):
            continue
        plt.scatter(X_pca[label_data == cluster][0], X_pca[label_data == cluster][1], label=label, c=color)
        if np.array_equal(label_data, label_hiercl):
            plt.scatter(pca[cluster-1][0], pca[cluster-1][1], label=label_centroid, c=colors[i], marker='X', s=250)
        else:
            plt.scatter(pca[cluster][0], pca[cluster][1], label=label_centroid, c=colors[i], marker='X', s=250)
        i+=1
    plt.legend()
    plt.show()


data_gaussian = np.load('processed-data/user-feature-set-gaussian.npy')

#  ΤΟ y_bsas ΠΕΡΙΕΧΕΙ ΤΙΣ ΕΤΙΚΕΤΕΣ ΟΜΑΔΟΠΟΙΗΣΗΣ ΒΑΣΗ ΤΟΥ bsas ΑΛΓΟΡΙΘΜΟΥ
label_bsas = np.load('processed-data/BSAS-data/clusters-gaussian.npy')
# ΤΟ y_kmeans ΠΕΡΙΕΧΕΙ ΤΙΣ ΕΤΙΚΕΤΕΣ ΟΜΑΔΟΠΟΙΗΣΗ ΒΑΣΗ ΤΟΥ k-means ΑΛΓΟΡΙΘΜΟΥ
label_kmeans = np.load('processed-data/clusters.npy')
#ΤΟ y_hiercl ΠΕΡΙΕΧΕΙ ΤΙΣ
label_hiercl = np.load('processed-data/hierarchical_clusters.npy')

#ΤΟ pca ΕΙΝΑΙ ΓΙΑ ΤΑ ΓΡΑΦΗΜΑΤΑ ΠΟΥ ΘΑ ΓΙΝΟΥΝ PLOT
pca = pca_decom(n_components=2) #2 ΔΙΑΣΤΑΣΕΙΣ ΓΙΑ ΤΗΝ ΜΕΤΑΤΡΟΠΗ PCA
X_pca = pd.DataFrame(pca.fit_transform(data_gaussian))

#ΤΟ ΜΕΓΕΘΟΣ ΤΟΥ ΠΙΝΑΚΑ
plt.figure(0, figsize=(25, 10))
plt.title('PCA RESULT PLOT')
plt.xlabel('X')
plt.ylabel('Y')

plt.scatter(X_pca[0], X_pca[1], color='black')

plt.show()

plot_pca_cluster(type='BSAS',label_data=label_bsas)
plot_pca_cluster(type='K-MEANS',label_data=label_kmeans)
plot_pca_cluster(type='HIERARCHICAL',label_data=label_hiercl)

