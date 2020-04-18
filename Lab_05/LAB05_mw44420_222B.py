import pandas as pd
import numpy as np
import scipy.sparse
import math


# def unique(x):
#     x.sort()
#     result = []
#     n = len(x)
#     for i in range(n - 1):
#         if x[i] != x[i + 1]:
#             result.append(x[i])
#     result.append(x[n - 1])
#     return (result)


def freq(x, prob):
    xi = np.unique(x);
    # n = len(x)
    p = []
    ni = pd.value_counts(x);

    if (prob == True):
        for i in range(len(xi)):
            p.append(ni.array[i] / len(xi))
        return xi, p

    return xi, ni;


def freq2(x, y, prob):
    xi = np.unique(x);
    yi = np.unique(y)
    xyi = []
    nx = len(xi);
    ny = len(yi);


    for i in range(nx):
        for j in range(ny):
            xyi.append([xi[i], yi[j]])

    tmp = 0
    nxy = len(xyi)
    ni = []
    for i in range(nxy):
        for j in range(nx):
            if [x[j], y[j]] == xyi[i]:
                tmp +=  1
        ni.append(tmp)
        tmp = 0

    if prob == True:
        pi = []
        for i in range(len(ni)):
            pi.append(ni[i] / len(x))
        return xi, yi, xyi, pi

    return xi, yi, xyi, ni




def entropy(x):
    xi, pi = freq(x, prob=True);
    n = len(pi);
    sum = 0;
    for i in range(n):
        sum += pi[i] * math.log2(pi[i]);

    ent = -sum;
    return ent;


def entropyxy(x, y):
    xi, yi, xyi, pi = freq2(x, y, prob=True);
    n = len(xyi);
    sum = 0;


    for i in range(n):
        if pi[i] != 0:
            sum += pi[i] * math.log2(pi[i])

    ent = -sum
    return ent;

def funkcjaPomocnicza(x, y):
    xi, yi, xyi, pi = freq2(x, y, prob=True);
    sum=0;

    for i in range(len(pi)):
        if pi[i] != 0:
            sum += pi[i] * math.log2(pi[i]);

    hYxi = -sum;
    return hYxi;

def entropiaWarunkowa(x, y):
    xi, yi, xyi, pi = freq2(x, y, prob=True);
    x1, px = freq(x, prob=True);
    ent = 0;

    for j in range(len(px)):
        ent += px[j] * funkcjaPomocnicza(x,y);

    return ent


def informacjaWzajemna(x, y):
    hx = entropy(x);
    hy = entropy(y);
    hxy = entropyxy(x, y)
    inf = hx + hy - hxy;
    return inf;

def infogain(x,y):
    hy=entropy(y);
    entCond = entropiaWarunkowa(x, y);
    inf = hy - entCond;
    return inf;


######################
x = [1, 1, 1, 2, 3, 4]
y = [5, 5, 2, 4, 5, 9]
# zad1
# print(freq(x, prob = True))

# zad2
# print(freq2(x, y, prob=True))

# ZAD3
# hx = entropy(x);
# hy = entropy(y);
# hxy = entropyxy(x, y);
# print(hxy);
# inf = informacjaWzajemna(x, y);
# print(inf);
# print(entropiaWarunkowa(x,y));
# print(infogain(x,y));
#
# # ZAD4
# data = pd.read_csv('zoo.csv');
# print(infogain('type', 'animal')); # -0.27368437626202313
# print(infogain('type', 'hair')); #-15.509775004326936
# print(infogain('type', 'feathers')); #-8.0
# print(infogain('type', 'eggs')); #-24.0
# print(infogain('type', 'milk')); #-8.0
# print(infogain('type', 'airborne')); #-8.0
# print(infogain('type', 'aquatic'));
# print(infogain('type', 'predator'));
# print(infogain('type', 'toothed'));
# print(infogain('type', 'backbone'));
# print(infogain('type', 'breathes'));
# print(infogain('type', 'venomous'));
# print(infogain('type', 'fins'));
# print(infogain('type', 'legs')); #-8.0
# print(infogain('type', 'tail')); #-8.0
# print(infogain('type', 'domestic')); # -24.0
# print(infogain('type', 'catsize')); #-19.651484454403228


### ZAD6
# Błędy podczas wczytywania bazy
# from sklearn.datasets import fetch_rcv1
#
# rcv1=fetch_rcv1()
# X=rcv1["data"]
# Y=rcv1.target[:,87]
