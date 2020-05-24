import cv2
import numpy as np
import pandas as pd
import scipy
import sklearn
from matplotlib import pyplot as plt
import random
# import Image
from PIL import Image
from PIL import ImageOps
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from sklearn import metrics
from sklearn import datasets
from sklearn import mixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
# from sklearn.mixture import GMM

# zad1
iris = datasets.load_iris()
X = iris.data
Y = iris.target

# zad2
# metoda najblizszego sasiedztwa:
single = AgglomerativeClustering(linkage='single', n_clusters=3)
single.fit(X);
# metoda srednich połaczen:
av = AgglomerativeClustering(linkage='average', n_clusters=3)
av.fit(X);
# metoda najdalszych połaczen:
complete = AgglomerativeClustering(linkage='complete', n_clusters=3)
complete.fit(X);
# metoda Warda:
ward = AgglomerativeClustering(linkage='ward', n_clusters=3)
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
### average=None -> dla każdej klasy (klastru) jest liczone

### zad5
pca = PCA(n_components=2)
X_r = pca.fit_transform(X)

mode = ['single', 'average', 'complete', 'ward']
Y_result = find_perm(3, Y, AgglomerativeClustering(linkage=mode[0], n_clusters=3).fit(X).labels_)
Y_result2 = AgglomerativeClustering(linkage=mode[2], n_clusters=3).fit(X).labels_

colors = ['navy', 'turquoise', 'darkorange']

plt.figure()
plt.subplot(3, 1, 1)
for color, i in zip(colors, [0, 1, 2]):
    points = X_r[Y == i, :]
    plt.scatter(points[:, 0], points[:, 1], color=color)
    hull = ConvexHull(points)
    plt.plot(points[hull.vertices, 0], points[hull.vertices, 1], color=color)

plt.subplot(3, 1, 2)
for color, i in zip(colors, [0, 1, 2]):
    points = X_r[Y_result2 == i, :]
    plt.scatter(points[:, 0], points[:, 1], color=color)
    if len(points):
        hull = ConvexHull(points)
        plt.plot(points[hull.vertices, 0], points[hull.vertices, 1], color=color)

plt.subplot(3, 1, 3)
correct = [Y[i] == Y_result2[i] for i in range(len(Y))]
incorrect = np.invert(correct)
plt.scatter(X_r[correct, 0], X_r[correct, 1], color='green')
plt.scatter(X_r[incorrect, 0], X_r[incorrect, 1], color='red')

plt.show()

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
#
# ### ZAD7
Z = scipy.cluster.hierarchy.linkage(X_r, 'average')
dn = dendrogram(Z)
plt.show();

### ZAD8
### k-means:
# pca = PCA(n_components=2)
# X_r = pca.fit_transform(X)
#
# mode = ['single', 'average', 'complete', 'ward']
# Y_resul = find_perm(3, Y, sklearn.cluster.KMeans(3).fit(X).labels_)
# Y_result = sklearn.cluster.KMeans(3).fit(X).labels_
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
#     points = X_r[Y_result == i, :]
#     plt.scatter(points[:, 0], points[:, 1], color=color)
#     if len(points):
#         hull = ConvexHull(points)
#         plt.plot(points[hull.vertices, 0], points[hull.vertices, 1], color=color)


# plt.subplot(3, 1, 3)
# correct = [Y[i] == Y_result[i] for i in range(len(Y))]
# incorrect = np.invert(correct)
# plt.scatter(X_r[correct, 0], X_r[correct, 1], color='green')
# plt.scatter(X_r[incorrect, 0], X_r[incorrect, 1], color='red')
#
# plt.show()

# zad6
# pca = PCA(n_components=3)
# X_r = pca.fit_transform(X)
#
# mode = ['single', 'average', 'complete', 'ward']
# Y_result = find_perm(3, Y, sklearn.cluster.KMeans(3).fit(X).labels_)
# Y_result2 = sklearn.cluster.KMeans(3).fit(X).labels_
#
# colors = ['navy', 'turquoise', 'darkorange']
#
# fig = plt.figure()
# a = fig.add_subplot(311, projection='3d')
# for color, i in zip(colors, [0, 1, 2]):
#     points = X_r[Y == i, :]
#     a.scatter(points[:, 0], points[:, 1], points[:, 2], color=color)
#
# a = fig.add_subplot(312, projection='3d')
# for color, i in zip(colors, [0, 1, 2]):
#     points = X_r[Y_result2 == i, :]
#     a.scatter(points[:, 0], points[:, 1], points[:, 2], color=color)
#
# a = fig.add_subplot(313, projection='3d')
# correct = [Y[i] == Y_result[i] for i in range(len(Y))]
# incorrect = np.invert(correct)
# a.scatter(X_r[correct, 0], X_r[correct, 1], X_r[correct, 2], color='green')
# a.scatter(X_r[incorrect, 0], X_r[incorrect, 1], X_r[incorrect, 2], color='red')
#
# plt.show()

### zad 9
#
# data = pd.read_csv('zoo.csv')
# X_zoo = data.values[:, 1:16]
#
# Y_result = AgglomerativeClustering(linkage='ward').fit(X).labels_
#
# print(Y_result)


################# 3. KWANTYZACJA ###############

### ZAD1
###rozmiar: 640x427
# im = Image.open("zdj.jpg")

# print(im.mode)

img = cv2.imread('zdj1.JPG')

###ZAD2
# print(img.shape);
w, k, col = img.shape;

tab = np.zeros((w*k, 3), int);
n = 0;
# print(w,k,col);
for i in range(w):
    for j in range(k):
        tab[n, :] = img[i,j,:]
        n=n+1;

# print(tab)

### ZAD3

###kmeans:
ile_klastrow = 8
k_means = sklearn.cluster.KMeans(ile_klastrow).fit(tab)
kmeans_labels = k_means.labels_
# print(kmeans_labels)
kmeans_centres = k_means.cluster_centers_
# print(kmeans_centres)

# print(kmeans_labels.shape)
# print(tab.shape)

###gmm:
gmm = sklearn.mixture.GaussianMixture(n_components=ile_klastrow).fit(tab)
gmm_labels = gmm.predict(tab)
gmm_centres = gmm.means_
# print(gmm_labels)
# print(gmm_centres)

### ZAD4
img_quant = np.zeros(tab.shape)
for i in range(len(kmeans_labels)):
    img_quant[i,:] = kmeans_centres[kmeans_labels[i]]
# print(img_quant)


img_quant_gmm = np.zeros(tab.shape)
for i in range(len(gmm_labels)):
    img_quant_gmm[i,:] = gmm_centres[gmm_labels[i]]

###ZAD5

n = 0;
tab1 = np.zeros((w, k , col), dtype=np.uint8)
for i in range(w):
    for j in range(k):
        tab1[i,j,:] = img_quant[n, :]
        n=n+1;


# tab_gmm = np.zeros((w, k , col), dtype=np.uint8)
# for i in range(w):
#     for j in range(k):
#         tab_gmm[i,j,:] = img_quant_gmm[n, :]
#         n=n+1;


###ZAD6

# print(tab1)
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(tab1)
plt.title("zad6")
plt.show()


### ZAD7
kmeans_blad = np.zeros((w, k))
for i in range (w):
    for j in range (k):
        kmeans_blad[i,j] = metrics.mean_squared_error(img[i,j], tab1[i,j])

plt.imshow(kmeans_blad)
plt.title("Bład sredniokwadratowy")
plt.show()



# gmm_blad = np.zeros((w, k))
# for i in range (w):
#     for j in range (k):
#         gmm_blad[i,j] = metrics.mean_squared_error(img[i,j], tab_gmm[i,j])

# plt.imshow(gmm_blad)
# plt.title("Bład sredniokwadratowy")
# plt.show()


###ZAD8
n = 2;
zad8_wektoryzacja = img.reshape((int)(w*(k/(2**n))),col*(2**n))
kmeans_zad8 = sklearn.cluster.KMeans(n_clusters = ile_klastrow).fit(zad8_wektoryzacja)
labels_kmeans_zad8 = kmeans_zad8.labels_
centers_kmeans_zad8 = kmeans_zad8.cluster_centers_

quant_kmeans_zad8 = np.copy(zad8_wektoryzacja)
for i in range(len(zad8_wektoryzacja)):
    quant_kmeans_zad8[i] = centers_kmeans_zad8[labels_kmeans_zad8[i]]

tab1 = quant_kmeans_zad8.reshape(w, k, col)
plt.imshow(tab1)
plt.title("zad8")
plt.show()
