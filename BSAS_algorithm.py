import pandas as pd

import matplotlib.pyplot as matplot
import numpy as np
from scipy.spatial.distance import euclidean
from tqdm import tqdm



class BSAS_model:
    # The constructor of an BSAS object
    def __init__(self, theta=None, q=None):
        # theta: the dissimilarity threshold
        # q: the max number of clusters
        self.theta = theta
        self.q = q

        # Our list of the clusters and
        # centroids that will help us later
        self.clusters = {}
        self.centroids = {}

    def calc_BSAS_data(self, data, n_times=50, n_theta=50, load_precalculated=True):

        l, N = data.shape
        if not(load_precalculated):
            minDist, maxDist = self.euclideanDistance(data, N)
            np.save('processed-data/BSAS-data/euclidean-distances.npy',np.array([minDist, maxDist], dtype=np.float))
            print('saved: processed-data/BSAS-data/euclidean-distances.npy')
        else:
            minDist, maxDist = np.load('processed-data/BSAS-data/euclidean-distances.npy')
            print('loaded: processed-data/BSAS-data/euclidean-distances.npy')

        meanDist = (minDist + maxDist) / 2
        minimum_theta = 0.25 * meanDist
        maximum_theta = 1.75 * meanDist

        s = (maximum_theta - minimum_theta) / (n_theta - 1)
        print(minimum_theta, maximum_theta, s)
        if not (load_precalculated):
            total_clusters = []
            total_order = []
            total_theta = np.arange(minimum_theta, maximum_theta + s, s)
            for theta in tqdm(total_theta, desc=('Please wait until the BSAS algorithm is complete --->')):

                max_clusters = -np.inf
                for i in np.arange(n_times):
                    clf = BSAS_model(theta=theta, q=N)
                    order = np.random.permutation(N)
                    clf.run_BSAS(data, order)
                    clusters, centroids = clf.predict()
                    clusterCount = len(clusters)

                    if (clusterCount > max_clusters):
                        max_clusters = clusterCount
                        order_max = order
                total_order = total_order + [order_max]
                total_clusters = total_clusters + [max_clusters]

            np.save('processed-data/BSAS-data/all_clusters-gaussian.npy', np.array(total_clusters, dtype=np.int))
            print('saved: processed-data/BSAS-data/all_clusters-gaussian.npy')
            np.save('processed-data/BSAS-data/order-gaussian.npy', np.array(total_order))
            print('saved: processed-data/BSAS-data/order-gaussian.npy')
            np.save('processed-data/BSAS-data/all_theta-gaussian.npy', np.array(total_theta, dtype=np.float))
            print('saved: processed-data/BSAS-data/all_theta-gaussian.npy')
        else:
            total_clusters = np.load('processed-data/BSAS-data/all_clusters-gaussian.npy')
            print('loaded: processed-data/BSAS-data/all_clusters-gaussian.npy')
            total_theta = np.load('processed-data/BSAS-data/all_theta-gaussian.npy')
            print('loaded: processed-data/BSAS-data/all_theta-gaussian.npy')


        matplot.plot(total_theta, total_clusters, 'b-')

        matplot.xlabel('Theta')
        matplot.ylabel('Number of clusters')
        matplot.title('Numbers clusters in corellation to theta')
        matplot.grid()
        matplot.show()

        opt_cluster = self.bestClusterNumber(total_clusters)
        print("The optimal cluster number is: %s"%(opt_cluster))
        opt_theta = self.bestTheta(opt_cluster, total_clusters, total_theta)
        print("The optimal theta is: %s" % (opt_theta))

        self.theta = opt_theta
        self.q = opt_cluster


    def run_BSAS(self, data, order):
        cluster_count = 1  # Count of cluster/centroids
        clusters = {}
        centroids = {}

        sample_one = data[:, order[0]]
        clusters[cluster_count - 1] = sample_one
        centroids[cluster_count - 1] = np.add(np.zeros_like(sample_one), sample_one)

        l, N = data.shape
        for i in range(1, N):
            sample = data[:, order[i]]
            distance, index = self.closestCluster(clusters, centroids, sample)
            if ((distance > self.theta) and (cluster_count < self.q)):
                cluster_count += 1
                clusters[cluster_count - 1] = sample
                centroids[cluster_count - 1] = np.add(np.zeros_like(sample), sample)
            else:
                clusters[index] = np.vstack((clusters[index], sample))
                centroids[index] = np.add(centroids[index], sample)


        self.clusters = clusters
        self.centroids = centroids



    def closestCluster(self, clusters, centroids, sample):
        centroid_id = 0
        cluster_count = clusters[centroid_id].shape
        centroid = self.getCentroid(centroids[centroid_id], cluster_count)

        minDist = euclidean(centroid, sample)
        try:
            for ID in centroids:
                if (ID == 0):
                    continue
                cluster_count = clusters[ID].shape
                centroid = self.getCentroid(centroids[ID], cluster_count)
                distance = euclidean(centroid, sample)
                if (distance < minDist):
                    minDist = distance
                    centroid_id = ID
        except:
            pass
        return minDist, centroid_id

    def euclideanDistance(self, data, size):
        min_euclidean = np.inf
        max_euclidean = -np.inf

        for i in tqdm(range(size), desc='Caclulating the euclidean distances of the dataset, please wait...-->'):
            for j in range(size):
                if (i == j):
                    continue #because we want j to be i+1
                distance = euclidean(data[:, i], data[:, j])
                if (distance < min_euclidean):
                    min_euclidean = distance
                if (distance > max_euclidean):
                    max_euclidean = distance

        return min_euclidean, max_euclidean


    def bestClusterNumber(self, clusters):
        cluster_appereances = {}
        smallest_cluster = np.min(clusters)
        for cluster_number in tqdm(clusters, desc='Caclulating the best number of cluster, please wait...-->'):
            if (cluster_number == smallest_cluster):
                continue
            try:
                cluster_appereances[cluster_number] += 1
            except:
                cluster_appereances[cluster_number] = 1

        best_cluster = None;
        best_cluster_appereances = -np.inf

        for index in cluster_appereances:
            tmp = cluster_appereances[index]
            if (tmp > best_cluster_appereances):
                best_cluster_appereances = tmp
                best_cluster = index
        return best_cluster

    def bestTheta(self, bestCluster, clusters, theta):

        cluster_startpoint = 0
        cluster_endpoint = 0


        found = False
        range_list = {}
        for i in range(len(clusters)):
            if clusters[i] == bestCluster:
                if not found:
                    cluster_startpoint = i
                    cluster_endpoint = i
                    found = True
                else:
                    cluster_endpoint += 1
            else:
                if (found):
                    tmp = [cluster_startpoint, cluster_endpoint, (cluster_endpoint - cluster_startpoint)]
                    range_list[i] = tmp

        cluster_index_key = None
        for index in range_list:
            max_range = -np.inf
            differ = range_list[index][2]
            if (differ > max_range):
                max_range = differ
                cluster_index_key = index

        bestTheta_range = range_list[cluster_index_key]
        averageTheta = 0
        for i in range(bestTheta_range[0], bestTheta_range[1] + 1):
            averageTheta += theta[i]

        averageTheta = averageTheta / (bestTheta_range[1] - bestTheta_range[0] + 1)
        return averageTheta


    def predict(self):
        predicted_centroids = {}
        for index in self.clusters:
            predicted_centroids[index] = self.getCentroid(self.centroids[index], self.clusters[index].shape)

        return self.clusters, predicted_centroids

    def param(self):
        return self.theta, self.q

    def getCentroid(self, centroid, cluster_count):
        try:
            null_checker = cluster_count[1]
            return np.divide(centroid, cluster_count[0])
        except:
            return centroid


