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
                total_order = [total_order] + [order_max]
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

        opt_cluster = self.findOptimalCluster(total_clusters)
        print("The optimal cluster number is: %s"%(opt_cluster))
        opt_theta = self.findOptimalTheta(opt_cluster, total_clusters, total_theta)
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


    def predict(self):
        predicted_centroids = {}
        for key in self.clusters:
            predicted_centroids[key] = self.getCentroid(self.centroids[key], self.clusters[key].shape)

        return self.clusters, predicted_centroids

    def param(self):
        return self.theta, self.q

    def getCentroid(self, centroid, cluster_count):
        try:
            probe = cluster_count[1]
            return np.divide(centroid, cluster_count[0])
        except:
            return centroid

    def closestCluster(self, clusters, centroids, sample):
        centID = 0
        cluster_population = clusters[centID].shape
        centroid = self.getCentroid(centroids[centID], cluster_population)

        minDist = euclidean(centroid, sample)
        try:
            for cntID in centroids:
                if (cntID == 0):
                    continue
                cluster_population = clusters[cntID].shape
                centroid = self.getCentroid(centroids[cntID], cluster_population)
                tmp = euclidean(centroid, sample)
                if (tmp < minDist):
                    minDist = tmp
                    centID = cntID
        except:
            pass
        return minDist, centID

    def euclideanDistance(self, data, size):
        minED = np.inf
        maxED = -np.inf

        for column_i in tqdm(range(size), desc='Computing (Min/Max) Euclidean Distances...'):
            for column_j in range(size):
                if (column_i == column_j):
                    continue
                dist = euclidean(data[:, column_i], data[:, column_j])
                if (dist < minED):
                    minED = dist
                if (dist > maxED):
                    maxED = dist

        return minED, maxED


    def findOptimalCluster(self, clusters):
        clusters_frq = {}
        min_cluster = np.min(clusters)
        for cluster in tqdm(clusters, desc='Finding Optimal Cluster...'):
            if (cluster == min_cluster):
                continue
            try:
                clusters_frq[cluster] += 1
            except:
                clusters_frq[cluster] = 1
        opt_cluster = None;
        frq_opt_cluster = -np.inf

        for key in clusters_frq:
            tmp = clusters_frq[key]
            if (tmp > frq_opt_cluster):
                frq_opt_cluster = tmp
                opt_cluster = key
        return opt_cluster

    def findOptimalTheta(self, opt_cluster, clusters, theta):
        cl_start = 0;
        cl_fin = 0;
        cl_key = None
        found = False
        cl_ranges = {}

        for i in range(len(clusters)):
            if (clusters[i] == opt_cluster):
                if (not found):
                    cl_start = i
                    cl_fin = i
                    found = True
                else:
                    cl_fin += 1
            else:
                if (found):
                    tmp = [cl_start, cl_fin, (cl_fin - cl_start)]
                    cl_ranges[i] = tmp

        for key in cl_ranges:
            max_range = -np.inf
            val = cl_ranges[key][2]
            if (val > max_range):
                max_range = val
                cl_key = key

        opt_theta_range = cl_ranges[cl_key]
        theta_avg = 0
        for i in range(opt_theta_range[0], opt_theta_range[1] + 1):
            theta_avg += theta[i]

        theta_avg = theta_avg / (opt_theta_range[1] - opt_theta_range[0] + 1)
        return theta_avg





