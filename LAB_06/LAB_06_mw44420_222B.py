import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random

from sklearn import metrics


# def distp(X, C):
#    return np.sqrt(np.add.outer(np.sum(X*X, axis=1),
#                                np.sum(C*C,axis=1)) - 2 * X.dot(C.T));


# def distm(X, C, V):
#     odl_mah = metrics.pairwise.pairwise_distances(X, C, metric='mahalanobis')

def dist(P1, P2):
    return np.sqrt(((P1[0] - P2[0]) ** 2) + (P1[1] - P2[1]) ** 2)


center_1 = np.array([1, 1])
center_2 = np.array([5, 5])
center_3 = np.array([8, 1])

# Generate random data and center it to the three centers
data_1 = np.random.randn(200, 2) + center_1
data_2 = np.random.randn(200, 2) + center_2
data_3 = np.random.randn(200, 2) + center_3

data = np.concatenate((data_1, data_2, data_3), axis=0)

plt.scatter(data[:, 0], data[:, 1], s=7)


# plt.show()
# print(data);

def kmeans(data, k):
    startPoints = np.zeros((k, 2))
    # print(startPoints)
    newCenters = []
    wiersze = np.size(data, 0)
    # print(wiersze)
    for i in range(k):
        x1 = np.random.randint(0, wiersze)
        startPoints[i, :] = data[x1]


     # print(startPoints)
     # plt.scatter(data[:,0], data[:,1], s=7)
     # plt.scatter(startPoints[:,0], startPoints[:,1], marker='*', c='g', s=150)
    # plt.show()

    distances = np.zeros((wiersze, k));

    while startPoints != newCenters:

        for i in range(len(data)):
            for j in range(k):
                distances[i, j] = dist(data[i], startPoints[j])
        print(data)
        print(distances)

        clusters = []
        for i in range(k):
            clusters.append([])

        for i in range(len(distances)):
            min = np.min(distances[i, :])
            index_min = np.argmin(distances[i, :])
            clusters[index_min].append(data[i, :])

        print(clusters)

        for i in range(k):
            newCenters.append(np.mean(clusters[i], 0))

        if startPoints != newCenters:
            startPoints = newCenters;



kmeans(data, 3);
