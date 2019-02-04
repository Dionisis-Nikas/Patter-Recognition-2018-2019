import PyQt5
import matplotlib.pyplot as plt
from matplotlib import style;  style.use('ggplot')
get_ipython().magic('matplotlib qt')

from tqdm import tqdm
from tqdm import tqdm_notebook
import numpy as np
from sklearn.cluster import KMeans

#ΤΟ ΑΡΧΕΙΟ ΠΟΥ ΕΧΕΙ ΜΕΣΑ ΤΟ np.load ΕΙΝΑΙ ΤΟ ΑΡΧΕΙΟ ΠΟΥ ΘΑ ΦΟΡΤΩΣΟΥΜΕ ΜΕ ΤΙ ΑΛΛΑΓΕΣ ΠΟΥ ΕΧΟΥΝ ΓΙΝΕΙ ΑΠΟ ΤΟ bsas
#ΜΠΟΡΟΥΜΕ ΝΑ ΤΟ ΟΝΟΜΑΣΟΥΜΕ ΟΠΩΣ ΘΕΛΟΥΜΕ, ΑΠΛΑ ΤΟ ΑΦΗΝΩ ΕΤΣΙ ΓΙΑ ΝΑ ΤΟ ΒΡΙΣΚΟΥΜΕ ΤΩΡΑ ΠΙΟ ΕΥΚΟΛΑ

X = np.load('comp-data/1-preprocessing-comp-data/user-feature-set-stdscl.npy')

#ΑΡΧΙΚΟΠΟΙΗΣΗ ΤΟΥ k-means ΑΛΓΟΡΙΘΜΟΥ ΓΙΑ 2 ΟΜΑΔΟΠΟΙΗΣΕΙΣ (clusters)
#n_init ΠΟΣΕΣ ΦΟΡΕΣ ΘΑ ΤΡΕΞΕΙ Ο ΑΛΓΟΡΙΘΜΟΣ k-means ΜΕ ΔΙΑΦΟΡΕΤΙΚΑ ΚΕΝΤΡΟΕΙΔΗ
#n_jobs=-1 ΕΙΝΑΙ ΓΙΑ ΝΑ ΧΡΗΣΙΜΟΠΟΙΟΥΝΤΑΙ ΣΤΟ ΜΕΓΙΣΤΟ ΤΗΣ % ΟΛΟΙ ΟΙ ΠΥΡΗΝΕΣ ΤΟΥ ΣΥΣΤΗΜΑΤΟΣ
#precompute_distances ΥΠΟΛΟΓΙΖΕΙ ΤΙΣ ΑΠΟΣΤΑΣΕΙΣ ΠΟΥ ΥΠΑΡΧΟΥΝ !!
#random_state ΓΙΑ ΚΑΘΕ ΦΟΡΑ ΠΟΥ ΤΡΕΧΕΙ Ο ΑΛΓΟΡΙΘΜΟΣ ΔΕΝ ΕΧΟΥΜΕ ΟΡΙΣΕΙ ΣΤΑΘΕΡΟ random_state ΚΑΙ ΕΠΟΜΕΝΩΣ ΠΕΡΝΟΥΜΕ ΔΙΑΦΟΡΕΤΙΚΟ ΑΠΟΤΕΛΕΣΜΑ ΠΟΥ ΕΙΝΑΙ ΛΟΓΙΚΗ ΣΥΜΠΕΡΙΦΟΡΑ ΤΟΥ ΑΛΓΟΡΙΘΜΟΥ
#verbose ΕΙΝΑΙ ΓΙΑ ΤΗΝ ΧΡΗΣΗ ΤΟΥ multithreaded
kmeans = KMeans(clusters=4, n_init=20, n_jobs=-1, precompute_distances=True, randοm_state=0, verbose=2)
kmeans.fit(X)

kmeans.labels_

kmeans.cluster_centers_

kmeans.inertia_

np.save('comp-data/3a-k-means-comp-data/clusters.npy', kmeans.labels_) #ΟΝΟΜΑΣΙΑ ΤΟΥ ΑΡΧΕΙΟΥ "clustering"
np.save('comp-data/3a-k-means-comp-data/cluster_centers_.npy', kmeans.cluster_centers_)  #ΟΝΟΜΑΣΙΑ ΤΟΥ ΑΡΧΕΙΟΥ "centers"
