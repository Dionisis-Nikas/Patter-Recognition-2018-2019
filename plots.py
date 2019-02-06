
import matplotlib.pyplot as plt
from matplotlib import style;  style.use('ggplot')

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as pca_decom

X = np.load('processed-data/user-feature-set-stdscl.npy')

#  ΤΟ y_bsas ΠΕΡΙΕΧΕΙ ΤΙΣ ΕΤΙΚΕΤΕΣ ΟΜΑΔΟΠΟΙΗΣΗΣ ΒΑΣΗ ΤΟΥ bsas ΑΛΓΟΡΙΘΜΟΥ
label_bsas = np.load('processed-data/BSAS-data/clusters-stdscl.npy')
# ΤΟ y_kmeans ΠΕΡΙΕΧΕΙ ΤΙΣ ΕΤΙΚΕΤΕΣ ΟΜΑΔΟΠΟΙΗΣΗ ΒΑΣΗ ΤΟΥ k-means ΑΛΓΟΡΙΘΜΟΥ
label_kmeans = np.load('processed-data/clusters.npy')
#ΤΟ y_hiercl ΠΕΡΙΕΧΕΙ ΤΙΣ
label_hiercl = np.load('processed-data/clusters_.npy')

#ΤΟ pca ΕΙΝΑΙ ΓΙΑ ΤΑ ΓΡΑΦΗΜΑΤΑ ΠΟΥ ΘΑ ΓΙΝΟΥΝ PLOT
pca = pca_decom(n_components=2) #2 ΔΙΑΣΤΑΣΕΙΣ ΓΙΑ ΤΗΝ ΜΕΤΑΤΡΟΠΗ PCA
X_pca = pd.DataFrame(pca.fit_transform(X))

#ΤΟ ΜΕΓΕΘΟΣ ΤΟΥ ΠΙΝΑΚΑ
plt.figure(0, figsize=(25, 10))
plt.title('PCA RESULT PLOT')
plt.xlabel('X')
plt.ylabel('Y')

plt.scatter(X_pca[0], X_pca[1], color='black')

plt.show()


tmp = pd.DataFrame(X_pca)
tmp[2] = label_bsas

c_bsas_pca = tmp.groupby([2]).mean()
c_bsas_pca = c_bsas_pca.values

plt.figure(1, figsize=(25, 10))
plt.title('PCA RESULT PLOT WITH BSAS ALGORITHM') #Ο ΤΙΤΛΟΣ
plt.xlabel('X')   # ΑΞΟΝΑΣ Χ
plt.ylabel('Y')   #ΑΞΟΝΑΣ Υ

#ΤΟ scatter ΧΡΗΣΙΜΟΠΟΙΕΤΑΙ ΓΙΑ ΝΑ ΣΚΟΡΠΙΣΕΙ ΤΑ ΣΤΟΙΧΕΙΑ ΚΑΙ ΝΑ ΤΑ ΒΑΛΕΙ ΣΕ ΕΝΑ ΓΡΑΦΗΜΑ
plt.scatter(X_pca[label_bsas==0][0], X_pca[label_bsas==0][1], label='Class 1', c='red')
plt.scatter(X_pca[label_bsas==1][0], X_pca[label_bsas==1][1], label='Class 2', c='blue')
plt.scatter(X_pca[label_bsas==2][0], X_pca[label_bsas==2][1], label='Class 3', c='lightgreen')
plt.scatter(X_pca[label_bsas==3][0], X_pca[label_bsas==3][1], label='Class 4', c='magenta')

plt.scatter(c_bsas_pca[0][0], c_bsas_pca[0][1], label='Class 1 Centroid', c='darkred', marker='X', s=200)
plt.scatter(c_bsas_pca[1][0], c_bsas_pca[1][1], label='Class 2 Centroid', c='darkblue', marker='X', s=200)
plt.scatter(c_bsas_pca[2][0], c_bsas_pca[2][1], label='Class 3 Centroid', c='darkgreen', marker='X', s=200)
plt.scatter(c_bsas_pca[3][0], c_bsas_pca[3][1], label='Class 4 Centroid', c='darkmagenta', marker='X', s=200)
plt.legend()
plt.show()

#Η ΙΔΙΑ ΛΟΓΙΚΗ ΑΚΟΛΟΥΘΕΙ ΚΑΙ ΓΙΑ ΤΟΥΣ ΥΠΟΛΟΙΠΟΥΣ ΥΠΟΛΟΓΙΣΜΟΥΣ ΚΑΙ ΕΜΦΑΝΙΣΕΙΣ ΤΩΝ ΓΡΑΦΗΜΑΤΩΝ
#ΕΠΟΜΕΝΩΣ ΓΙΝΟΝΤΑΙ 3 ΕΜΦΑΝΙΣΕΙΣ ΠΙΝΑΚΩΝ ΜΕ ΓΡΑΦΗΜΑΤΑ Ο ΚΑΘΕ ΕΝΑΣ ΓΙΑ

tmp = pd.DataFrame(X_pca)
tmp[2] = label_kmeans

c_kmeans_pca = tmp.groupby([2]).mean()
c_kmeans_pca = c_kmeans_pca.values
c_kmeans_pca

plt.figure(2, figsize=(25, 10))
plt.title('PCA RESULT WITH K-MEANS ALGORITHM (K=4)')
plt.xlabel('X')
plt.ylabel('Y')

plt.scatter(X_pca[label_kmeans==0][0], X_pca[label_kmeans==0][1], label='Class 1', c='red')
plt.scatter(X_pca[label_kmeans==1][0], X_pca[label_kmeans==1][1], label='Class 2', c='blue')
plt.scatter(X_pca[label_kmeans==2][0], X_pca[label_kmeans==2][1], label='Class 3', c='lightgreen')
plt.scatter(X_pca[label_kmeans==3][0], X_pca[label_kmeans==3][1], label='Class 4', c='magenta')

plt.scatter(c_kmeans_pca[0][0], c_kmeans_pca[0][1], label='Class 1 Centroid', c='darkred', marker='X', s=200)
plt.scatter(c_kmeans_pca[1][0], c_kmeans_pca[1][1], label='Class 2 Centroid', c='darkblue', marker='X', s=200)
plt.scatter(c_kmeans_pca[2][0], c_kmeans_pca[2][1], label='Class 3 Centroid', c='darkgreen', marker='X', s=200)
plt.scatter(c_kmeans_pca[3][0], c_kmeans_pca[3][1], label='Class 4 Centroid', c='darkmagenta', marker='X', s=200)
plt.legend()
plt.show()

tmp = pd.DataFrame(X_pca)
tmp[2] = label_hiercl

c_hiercl_pca = tmp.groupby([2]).mean()
c_hiercl_pca = c_hiercl_pca.values
c_hiercl_pca

plt.figure(3, figsize=(25, 10))
plt.title('PCA RESULT WITH HIERARCHICAL ALGORITHM')
plt.xlabel('X')
plt.ylabel('Y')

plt.scatter(X_pca[label_hiercl==1][0], X_pca[label_hiercl==1][1], label='Class 1', c='red')
plt.scatter(X_pca[label_hiercl==2][0], X_pca[label_hiercl==2][1], label='Class 2', c='blue')
plt.scatter(X_pca[label_hiercl==3][0], X_pca[label_hiercl==3][1], label='Class 3', c='lightgreen')
plt.scatter(X_pca[label_hiercl==4][0], X_pca[label_hiercl==4][1], label='Class 4', c='magenta')

plt.scatter(c_hiercl_pca[0][0], c_hiercl_pca[0][1], label='Class 1 Centroid', c='darkred', marker='X', s=200)
plt.scatter(c_hiercl_pca[1][0], c_hiercl_pca[1][1], label='Class 2 Centroid', c='darkblue', marker='X', s=200)
plt.scatter(c_hiercl_pca[2][0], c_hiercl_pca[2][1], label='Class 3 Centroid', c ='darkgreen', marker='X', s=200)
plt.scatter(c_hiercl_pca[3][0], c_hiercl_pca[3][1], label='Class 4 Centroid', c='darkmagenta', marker='X', s=200)

plt.legend()
plt.show()