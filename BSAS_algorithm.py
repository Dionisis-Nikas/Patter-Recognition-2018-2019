import matplotlib.pyplot as matplot
import numpy as np
from scipy.spatial.distance import euclidean
from tqdm import tqdm


class BSAS:
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

    def param(self):
        return self.theta, self.q

    def getCentroid(self, X, Y):
        try:

            return np.divide(X, Y[0])
        except:
            return X

    def closetCluster(self, clusters, centroids, sample):
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

    def __getEuclideanDistances(self, data, size):
        minED = np.inf;
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

    def __findIndexofMax(self, dct):
        minVal = np.inf
        minKey = None
        for key in dct:
            tmp = dct[key]
            if (tmp < minVal):
                minVal = tmp
                minKey = key
        return minKey

    def __findOptimalCluster(self, clusters):
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

    def __findOptimalTheta(self, opt_cluster, clusters, theta):
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

    def fit(self, data, order):
        m = 1  # Count of cluster/centroids
        clusters = {}
        centroids = {}

        sample_one = data[:, order[0]]
        clusters[m - 1] = sample_one
        centroids[m - 1] = np.add(np.zeros_like(sample_one), sample_one)

        N, l = data.shape
        for i in range(1, l):
            sample = data[:, order[i]]
            dist, k = self.closetCluster(clusters, centroids, sample)
            if ((dist > self.theta) and (m < self.q)):
                m += 1
                clusters[m - 1] = sample
                centroids[m - 1] = np.add(np.zeros_like(sample), sample)
            else:
                clusters[k] = np.vstack((clusters[k], sample))
                centroids[k] = np.add(centroids[k], sample)

        self.clusters = clusters
        self.centroids = centroids

    def fit_best(self, data, n_times=20, n_theta=50, first_time=True, dataname=None, plot_graph=False):
        var = ('-' + dataname) if dataname is not None else ''
        N, l = data.shape
        if (first_time):
            minDist, maxDist = self.__getEuclideanDistances(data, l)
            dists = np.save('processed-data/BSAS-data/min-max-euclidean-distances%s.npy' % (var),
                            np.array([minDist, maxDist], dtype=np.float))
            print ('saved: processed-data/BSAS-data/min-max-euclidean-distances%s.npy' % (var))
        else:
            minDist, maxDist = np.load('processed-data/BSAS-data/min-max-euclidean-distances%s.npy' % (var))
            print ('loaded: processed-data/BSAS-data/min-max-euclidean-distances%s.npy' % (var))

        meanDist = (minDist + maxDist) / 2
        theta_min = 0.25 * meanDist
        theta_max = 0.75 * meanDist

        s = (theta_max - theta_min) / (n_theta - 1)
        print (theta_min,theta_max,s)

        if (first_time):
            total_clusters = []
            total_theta = np.arange(theta_min, theta_max + s, s)
            for theta in tqdm(total_theta, desc=('Running BSAS...')):

                max_clusters = -np.inf
                for i in np.arange(n_times):
                    clf = BSAS(theta=theta, q=l)
                    order = np.random.permutation(range(l))
                    clf.fit(data, order)
                    clusters, centroids = clf.predict()
                    clustersN = len(clusters)
                    if (clustersN > max_clusters):
                        max_clusters = clustersN

                total_clusters = total_clusters + [max_clusters]


            np.save('processed-data/BSAS-data/total_clusters%s.npy' % (var), np.array(total_clusters, dtype=np.int))
            print ('saved: processed-data/BSAS-data/total_clusters.npy')
            np.save('processed-data/BSAS-data/total_theta%s.npy' % (var), np.array(total_theta, dtype=np.float))
            print ('saved: processed-data/BSAS-data/total_theta.npy')
        else:
            total_clusters = np.load('processed-data/BSAS-data/total_clusters%s.npy' % (var))
            print ('loaded: processed-data/BSAS-data/total_clusters%s.npy' % (var))
            total_theta = np.load('processed-data/BSAS-data/total_theta%s.npy' % (var))
            print ('loaded: processed-data/BSAS-data/total_theta%s.npy' % (var))

        if (plot_graph == True):
            matplot.plot(total_theta, total_clusters, 'b-')
            print(total_clusters)
            matplot.xlabel('theta')
            matplot.ylabel('Nu. of clusters')
            matplot.title('Nu. clusters in corellation to theta')
            matplot.grid()
            matplot.show()

        opt_cluster = self.__findOptimalCluster(total_clusters)
        print (opt_cluster)
        opt_theta = self.__findOptimalTheta(opt_cluster, total_clusters, total_theta)
        print (opt_theta)

        self.theta = opt_theta
        self.q = opt_cluster

    def predict(self):
        real_centroids = {}
        for key in self.clusters:
            real_centroids[key] = self.getCentroid(self.centroids[key], self.clusters[key].shape)

        return self.clusters, real_centroids


