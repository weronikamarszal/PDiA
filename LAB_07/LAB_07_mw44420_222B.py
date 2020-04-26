import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
import random

from sklearn import metrics
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering


# zad1
iris = datasets.load_iris()
X = iris.data
Y = iris.target

# zad2
# metoda najblizszego sasiedztwa:
single = AgglomerativeClustering(linkage='single')
# metoda srednich połaczen:
avg = AgglomerativeClustering(linkage='avarange')
avg.fit(X);
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
    perm=[]
    for i in range(clusters):
        idx = Y_pred == i
        new_label=scipy.stats.mode(Y_real[idx])[0][0]
        perm.append(new_label)
    return [perm[label] for label in Y_pred]