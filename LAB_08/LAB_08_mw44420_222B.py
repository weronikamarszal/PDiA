import pandas as pd
import numpy as np
import scipy.sparse
from scipy.sparse import random
from scipy.sparse import coo_matrix
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
import math
import numpy.linalg as npl
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
#
# X = np.array([[2,1,3], [3,0,2], [1,3,1], [4,4,0]])
#
# def wiPCA(X):
#     # X = np.transpose(X)
#     # print(X)
#     P=np.size(X, 0)
#     # print(P)
#
#     X_transp = np.transpose(X)
#     # print(X_transp)
#
#     # macierz korelacji:
#     R =(np.dot( X_transp, X)) / P;
#     # print(R)
#
#     D,V =  npl.eig(R)
#     # print("V: ")
#     # print(V)
#     # print("D: ")
#     # print(D)
#     # print("")
#     D_diag = np.diag(D);
#     D_transp = D.transpose()
#     print()
#
#     d = np.sort(D)[::-1];
#     # print(d)
#     I = np.argsort(D)[::-1];
#     # print(I)
#
#     D_diag = np.diag(d);
#     # print(D_diag)
#
#     V_sorted = V[:, I];
#     # print(V_sorted)
#
#     K = 2; #ile wymiarow
#
#
#     W = np.transpose(V_sorted[:, 0:K])
#     # print(W)
#     # print(X)
#     X=np.transpose(X)
#     Y = np.dot(W,X_transp);
#     print(Y.shape)
#     # print(Y)
#
#     return Y
#
# iris = datasets.load_iris()
#
# # pca = PCA(n_components=2)
# # X_r = pca.fit(iris.data).transform(iris.data)

### ZADANIE 1.
### A
X = np.dot(np.random.rand(2, 2), np.random.rand(2, 100)).T

### B
plt.figure()
plt.scatter(X[:, 0], X[:, 1])
plt.title("losowe punkty")
plt.show()

# punkt C
def wiPCA(X, K):
    X_sr = np.mean(X, axis=0)
    X = X - X_sr
    macierzKowariancji = np.cov(X, rowvar=False)
    eigVal, eigVec = np.linalg.eigh(macierzKowariancji)
    index = np.argsort(eigVal)[::-1]
    eigVec = eigVec[:, index]
    eigVec = eigVec[:, 0:K]
    eigVal = eigVal[index]
    Y = np.dot(X, eigVec)
    Y_inv = np.dot(Y, np.transpose(eigVec)) + X_sr

    return Y, eigVec, eigVal, Y_inv, X_sr,


wiPCA1, eigVec, eigVal, wiPCA1_inv, meanVec = wiPCA(X, 1)
plt.scatter(X[:, 0], X[:, 1], c='green')
plt.scatter(wiPCA1_inv[:, 0], wiPCA1_inv[:, 1], c='red')
plt.title("wizualizacja przestrzeni cech i wektorów własnych")
plt.show()

### ZAD2
### A
iris = datasets.load_iris()
X = iris.data
Y = iris.target

### B
wiPCA2, eigVec, eigVal, wiPCA2_inv, meanVec = wiPCA(X, 2)
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

### C
plt.scatter(X_r[:, 0], X_r[:, 1], c=Y)
plt.title("Iris - pca")
plt.show()

plt.scatter(wiPCA2[:, 0], wiPCA2[:, 1], c=Y)
plt.title("Iris - wiPCA")
plt.show()

### ZAD3
### A
digits = datasets.load_digits()
X = digits.data
Y = digits.target

### B
wiPCA2, eigVec, eigVal, wiPCA2_inv, meanVec = wiPCA(X, 2)

### C
krzywa_wariancji = np.cumsum(eigVal / eigVal.sum())
plt.plot(krzywa_wariancji)
plt.title("krzywa wariancji")
plt.show()

### D
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)
plt.scatter(X_r[:, 0], X_r[:, 1], c=Y)
plt.title("Digits - PCA")
plt.show()

wiPCA2, eigVec, eigVal, wiPCA2_inv, meanVec = wiPCA(X, 2)
plt.scatter(wiPCA2[:, 0], wiPCA2[:, 1], c=Y)
plt.title("digits - wiPCA")
plt.show()
