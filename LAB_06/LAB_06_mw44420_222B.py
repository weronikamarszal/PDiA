import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random

from sklearn import metrics, datasets
from sklearn.decomposition import PCA


def kmeans(data_, k):
    newCenters = np.zeros((k, 2))
    oldCenters = np.zeros((k,2))
    wiersze = np.size(data_, 0)

    for i in range(k):
        x1 = np.random.randint(0, wiersze)
        newCenters[i, :] = data_[x1]

    distances = np.zeros((wiersze, k));
    leastDistance = np.zeros((wiersze, 1));

    while (np.array_equal(newCenters, oldCenters) == False):
        oldCenters = newCenters;

        for i in range(len(data_)):
            for j in range(k):
                distances[i, j] = dist(data_[i], newCenters[j])

        clusterForColoring = np.zeros(wiersze)
        clusters = []
        for i in range(k):
            clusters.append([])


        for i in range(len(distances)):
            index_min = np.argmin(distances[i, :])
            leastDistance[i] = distances[i, index_min]
            clusters[index_min].append(data_[i, :])
            clusterForColoring[i] = index_min;

        for i in range(k):
            newCenters[i, :] = np.mean(clusters[i], 0)

        distanceSum = leastDistance.sum()


    return clusterForColoring, newCenters, distanceSum;

def kmeansBest(data__, k):
    tries = 10;
    distMin = 0;
    distances = np.zeros((tries,1));
    for i in range(tries):
        cl, cn, dist = kmeans(data__, k)
        distances[i] = dist;
        print(i)
        if ((dist < distMin) or distMin==0):
            distMin = dist;
            clMin = cl;
            cnMin = cn;

    return clMin, cnMin, distances;

data = pd.read_csv('autos.csv');
data = pd.get_dummies(data).to_numpy();
data = data[~np.isnan(data).any(axis=1)]
pca = PCA(n_components=2)
data = pca.fit_transform(data)
# print(data)


def dist(P1, P2):
    return np.sqrt(((P1[0] - P2[0]) ** 2) + (P1[1] - P2[1]) ** 2)


# center_1 = np.array([1, 1])
# center_2 = np.array([5, 5])
# center_3 = np.array([8, 1])
#
# # Generate random data and center it to the three centers
# data_1 = np.random.randn(200, 2) + center_1
# data_2 = np.random.randn(200, 2) + center_2
# data_3 = np.random.randn(200, 2) + center_3
#
# data = np.concatenate((data_1, data_2, data_3), axis=0)

iris = datasets.load_iris()
X = iris.data
Y = iris.target
w, k = X.shape
pca = PCA(n_components=2);
X = pca.fit_transform(X)
data = X



cl, cent, distances = kmeansBest(data, 3);

print(distances);

plt.figure()
plt.plot(distances.transpose()[0,:])
plt.show()

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=cl)
plt.scatter(cent[:, 0], cent[:, 1], marker='*', c='g', s=150)
plt.show()




