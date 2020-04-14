import pandas as pd
import numpy as np
import scipy.sparse

data = pd.read_csv('zoo.csv');


def unique(x):
    x.sort()
    result = []
    n = len(x)
    for i in range(n - 1):
        if x[i] != x[i + 1]:
            result.append(x[i])
    result.append(x[n - 1])
    return (result)


def freq(x, prob):
    xi = unique(x);
    # n = len(x)
    p = []
    ni = pd.value_counts(x);

    if (prob == True):
        for i in range(len(xi)):
            p.append(ni.array[i]/len(xi))
        return xi, p

    return xi, ni;

def freq2(x, y, p):
    xi = unique(x);
    yi = unique(y)
    x1 = x + y;
    u, ni = freq(x1, prob=False);
    if (p == True):
        u1, p = freq(x1, prob=True);
        return xi, yi, p;
    return xi, yi, ni;


x = [1, 1, 1, 2, 3, 4]
y = [5, 5, 2, 4, 5, 9]
# zad1
# print(freq(x, prob = True))

# zad2
print(freq2(x, y, p=False))
