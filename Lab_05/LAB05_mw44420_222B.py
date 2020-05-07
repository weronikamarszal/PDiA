import pandas as pd
import numpy as np
import scipy.sparse
from scipy.sparse import random
from scipy.sparse import coo_matrix
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
import math

data = pd.read_csv('zoo.csv');

def freq(x, prob):
    xi = np.unique(x);
    # n = len(x)
    p = []
    ni = pd.value_counts(x);

    if (prob == True):
        for i in range(len(ni)):
            p.append(ni.array[i] / len(x))
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
        if pi[i] != 0:
            sum += pi[i] * math.log2(pi[i]);

    ent = -sum;
    return ent;


def entropyxy(x, y):
    xi, yi, xyi, pi = freq2(x, y, prob=True);
    n = len(pi);
    ent = 0;


    for i in range(n):
        if pi[i] != 0:
            ent += pi[i] * math.log2(1/pi[i])

    return ent;


# def funkcjaPomocnicza(x, y):
#     xi, yi, xyi, pi = freq2(x, y, prob=True);
#     sum=0;
#
#     for i in range(len(pi)):
#         if pi[i] != 0:
#             sum += pi[i] * math.log2(pi[i]);
#
#     hYxi = -sum;
#     return hYxi;
#
# def entropiaWarunkowa(x, y):
#     xi, yi, xyi, pi = freq2(x, y, prob=True);
#     x1, px = freq(x, prob=True);
#     ent = 0;
#
#     for j in range(len(px)):
#         ent += px[j] * funkcjaPomocnicza(x,y);
#
#     return ent


def informacjaWzajemna(x, y):
    hx = entropy(x);
    hxy = entropyxy(y, x);
    inf = hx - hxy;
    return inf;

# def infogain(x,y):
#     hy=entropy(y);
#     entCond = entropiaWarunkowa(x, y);
#     inf = hy - entCond;
#     return inf;


######################
x = [1, 1, 1, 2, 3, 4]
y = [5, 5, 2, 4, 5, 9]



# zad1
# print(freq(x, prob = True))
xi, ni = freq(data['legs'], prob=False)
print("xi, ni:")
print(xi, ni)


# zad2
# print(freq2(x, y, prob=False))
xi, yi, xy, pi = freq2(data['legs'], data['hair'], prob=True)
print("xi:")
print(xi)
print("yi:")
print(yi)
print("xy:")
print(xy)
print("pi:")
print(pi)


# ZAD3
# hx = entropy(x);
# hy = entropy(y);
# hxy = entropyxy(x, y);
# print(hxy);
# inf = informacjaWzajemna(x, y);
# print(inf);

print("entropia")
print(entropy(data['tail']));

print("ent warunkowa")
print(entropyxy(data['tail'], data['hair']))

print("przyrost inf")
print(informacjaWzajemna(data['tail'], data['hair']))

#
# # ZAD4
print("zad4 - inf wzajemna:")
print(informacjaWzajemna( data['animal'], data['type'])); #6.176949300778882
print(informacjaWzajemna( data['hair'], data['type'])); #0.6761323981525371
print(informacjaWzajemna( data['feathers'], data['type'])); #0.41005190348112663
print(informacjaWzajemna( data['eggs'], data['type'])); #0.6715681456825652
print(informacjaWzajemna( data['milk'], data['type'])); #0.6664216480905257
print(informacjaWzajemna( data['airborne'], data['type'])); #0.4831682250710939
print(informacjaWzajemna( data['aquatic'], data['type'])); #0.6317865988536917
print(informacjaWzajemna( data['predator'], data['type'])); #0.6241226674547967
print(informacjaWzajemna( data['toothed'], data['type'])); #0.660688643526387
print(informacjaWzajemna( data['backbone'], data['type'])); #0.3682646688637552
print(informacjaWzajemna( data['breathes'], data['type'])); #0.429591494194581
print(informacjaWzajemna( data['venomous'], data['type'])); #0.09148400940543289
print(informacjaWzajemna( data['fins'], data['type'])); #0.3459418076071684
print(informacjaWzajemna( data['legs'], data['type'])); #1.7259132710449587
print(informacjaWzajemna( data['tail'], data['type'])); #0.46038658336444793
print(informacjaWzajemna( data['domestic'], data['type'])); #0.20738436710088282
print(informacjaWzajemna( data['catsize'], data['type'])); #0.6801181421343

