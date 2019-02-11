import numpy as np
from sklearn.cluster import KMeans

#ΤΟ ΑΡΧΕΙΟ ΠΟΥ ΕΧΕΙ ΜΕΣΑ ΤΟ np.load ΕΙΝΑΙ ΤΟ ΑΡΧΕΙΟ ΠΟΥ ΘΑ ΦΟΡΤΩΣΟΥΜΕ ΜΕ ΤΙ ΑΛΛΑΓΕΣ ΠΟΥ ΕΧΟΥΝ ΓΙΝΕΙ ΑΠΟ ΤΟ bsas
#ΜΠΟΡΟΥΜΕ ΝΑ ΤΟ ΟΝΟΜΑΣΟΥΜΕ ΟΠΩΣ ΘΕΛΟΥΜΕ, ΑΠΛΑ ΤΟ ΑΦΗΝΩ ΕΤΣΙ ΓΙΑ ΝΑ ΤΟ ΒΡΙΣΚΟΥΜΕ ΤΩΡΑ ΠΙΟ ΕΥΚΟΛΑ

features = np.load('processed-data/gaussian-dataset.npy')

#ΑΡΧΙΚΟΠΟΙΗΣΗ ΤΟΥ k-means ΑΛΓΟΡΙΘΜΟΥ ΓΙΑ 2 ΟΜΑΔΟΠΟΙΗΣΕΙΣ (clusters)
#n_init ΠΟΣΕΣ ΦΟΡΕΣ ΘΑ ΤΡΕΞΕΙ Ο ΑΛΓΟΡΙΘΜΟΣ k-means ΜΕ ΔΙΑΦΟΡΕΤΙΚΑ ΚΕΝΤΡΟΕΙΔΗ
#precompute_distances ΥΠΟΛΟΓΙΖΕΙ ΤΙΣ ΑΠΟΣΤΑΣΕΙΣ ΠΟΥ ΥΠΑΡΧΟΥΝ !!
#random_state ΓΙΑ ΚΑΘΕ ΦΟΡΑ ΠΟΥ ΤΡΕΧΕΙ Ο ΑΛΓΟΡΙΘΜΟΣ ΔΕΝ ΕΧΟΥΜΕ ΟΡΙΣΕΙ ΣΤΑΘΕΡΟ random_state ΚΑΙ ΕΠΟΜΕΝΩΣ ΠΕΡΝΟΥΜΕ ΔΙΑΦΟΡΕΤΙΚΟ ΑΠΟΤΕΛΕΣΜΑ ΠΟΥ ΕΙΝΑΙ ΛΟΓΙΚΗ ΣΥΜΠΕΡΙΦΟΡΑ ΤΟΥ ΑΛΓΟΡΙΘΜΟΥ
#verbose ΕΙΝΑΙ ΓΙΑ ΤΗΝ ΧΡΗΣΗ ΤΟΥ multithreaded
kmeans = KMeans(n_clusters=4, n_init=20, precompute_distances=True, random_state=0, verbose=2)
kmeans.fit(features)

print(kmeans.labels_)

print(kmeans.cluster_centers_)

print(kmeans.inertia_)

np.save('processed-data/clusters.npy', kmeans.labels_) #ΟΝΟΜΑΣΙΑ ΤΟΥ ΑΡΧΕΙΟΥ "clustering"
np.save('processed-data/cluster_centers_.npy', kmeans.cluster_centers_)  #ΟΝΟΜΑΣΙΑ ΤΟΥ ΑΡΧΕΙΟΥ "centers"
