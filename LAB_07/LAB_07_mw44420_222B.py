import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
import random

from scipy.cluster.hierarchy import dendrogram
from scipy.spatial import ConvexHull, convex_hull_plot_2d

from sklearn import metrics
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

# zad1
iris = datasets.load_iris()
X = iris.data
Y = iris.target
# print(X)
# print(X.size)
# print(Y)


# zad2
# metoda najblizszego sasiedztwa:
single = AgglomerativeClustering(linkage='single')
single.fit(X);
# metoda srednich połaczen:
av = AgglomerativeClustering(linkage='average')
av.fit(X);
# metoda najdalszych połaczen:
complete = AgglomerativeClustering(linkage='complete')
complete.fit(X);
# metoda Warda:
ward = AgglomerativeClustering(linkage='ward')
ward.fit(X);


# metody aglomeracyjne - każda obserwacja tworzy na początku jednoelementowy klaster.
# Następnie pary klastrów są scalane,
# w każdej iteracji algorytmu łączone są ze sobą dwa najbardziej zbliżone klastry.

# zad3
def find_perm(clusters, Y_real, Y_pred):
    perm = []
    for i in range(clusters):
        idx = Y_pred == i
        new_label = scipy.stats.mode(Y_real[idx])[0][0]
        perm.append(new_label)
    return [perm[label] for label in Y_pred]


# print(single.labels_)
#
# x = find_perm(2, Y, single.labels_);
# print(x)

# Funkcja wyszukuje najczęstsze elementy każdego klastra. Wynikiem jest macierz,
# której każdy element jest najcześciej występującym elementem danego klastra.

# zad4
# Współczynnik Jaccarda mierzy podobieństwo między dwoma zbiorami i jest zdefiniowany
# jako iloraz mocy części wspólnej zbiorów i mocy sumy tych zbiorów

# print(metrics.jaccard_score(Y, single.labels_, average=None));
# average=None -> dla każdej klasy (klastru) jest liczone

# zad5
# pca = PCA(n_components=2)
# X_r = pca.fit_transform(X)
#
# mode = ['single', 'average', 'complete', 'ward']
# Y_result = find_perm(2, Y, AgglomerativeClustering(linkage=mode[0]).fit(X).labels_)
# Y_result2 = AgglomerativeClustering(linkage=mode[2]).fit(X).labels_
#
# colors = ['navy', 'turquoise', 'darkorange']
#
# plt.figure()
# plt.subplot(3, 1, 1)
# for color, i in zip(colors, [0, 1, 2]):
#     points = X_r[Y == i, :]
#     plt.scatter(points[:, 0], points[:, 1], color=color)
#     hull = ConvexHull(points)
#     plt.plot(points[hull.vertices, 0], points[hull.vertices, 1], color=color)
#
# plt.subplot(3, 1, 2)
# for color, i in zip(colors, [0, 1, 2]):
#     points = X_r[Y_result2 == i, :]
#     plt.scatter(points[:, 0], points[:, 1], color=color)
#     if len(points):
#         hull = ConvexHull(points)
#         plt.plot(points[hull.vertices, 0], points[hull.vertices, 1], color=color)
#
# plt.subplot(3, 1, 3)
# correct = [Y[i] == Y_result2[i] for i in range(len(Y))]
# incorrect = np.invert(correct)
# plt.scatter(X_r[correct, 0], X_r[correct, 1], color='green')
# plt.scatter(X_r[incorrect, 0], X_r[incorrect, 1], color='red')

# plt.show()

# zad6
pca = PCA(n_components=3)
X_r = pca.fit_transform(X)

mode = ['single', 'average', 'complete', 'ward']
Y_result = find_perm(2, Y, AgglomerativeClustering(linkage=mode[0]).fit(X).labels_)
Y_result2 = AgglomerativeClustering(linkage=mode[2]).fit(X).labels_

# fig = plt.figure()
# fig.add_subplot(3, 1, 1, projection='3d')
# plt.scatter(X_r[:, 0], X_r[:, 1], X_r[:, 2], color='blue')

colors = ['navy', 'turquoise', 'darkorange']

fig = plt.figure()
a = fig.add_subplot(311, projection='3d')
for color, i in zip(colors, [0, 1, 2]):
    points = X_r[Y == i, :]
    a.scatter(points[:, 0], points[:, 1], points[:, 2], color=color)

a = fig.add_subplot(312, projection='3d')
for color, i in zip(colors, [0, 1, 2]):
    points = X_r[Y_result2 == i, :]
    a.scatter(points[:, 0], points[:, 1], points[:, 2], color=color)

a = fig.add_subplot(313, projection='3d')
correct = [Y[i] == Y_result2[i] for i in range(len(Y))]
incorrect = np.invert(correct)
a.scatter(X_r[correct, 0], X_r[correct, 1], X_r[correct, 2], color='green')
a.scatter(X_r[incorrect, 0], X_r[incorrect, 1], X_r[incorrect, 2], color='red')

plt.show()

### ZAD7
Z = scipy.cluster.hierarchy.linkage(X_r, 'average')
dn = dendrogram(Z)
plt.show();

