#Magdlena Rybarczyk
#gr. 222A
#Lab 04

import numpy as np
import math as mt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

#DYSKRETYZACJA
#1-2-3

f = 10
Fs = 20
def dyskretyzacja(f, Fs):
    t0 = 0
    tn = 0
    tN = 1
    dt = 1.0/Fs
    n = 0

    s = []
    t = []

    while tn < tN:
        s.append(mt.sin(2*mt.pi*f*tn))
        t.append(tn)
        tn = tn + dt
        n = n + 1

    return t, s

#częstotliwość f = 10 Hz, próbkowanie Fs = 20 Hz
f = 10
Fs = 20
a, b = dyskretyzacja(f, Fs)
plt.plot(a, b)
plt.xlabel('t')
plt.ylabel('s(t)')
plt.title('f = 10[Hz], Fs = 20[Hz]')
plt.show()

#częstotliwość f = 10 Hz, próbkowanie Fs = 21 Hz
f = 10
Fs = 21
a, b = dyskretyzacja(f, Fs)
plt.plot(a, b)
plt.xlabel('t')
plt.ylabel('s(t)')
plt.title('f = 10[Hz], Fs = 21[Hz]')
plt.show()

#częstotliwość f = 10 Hz, próbkowanie Fs = 30 Hz
f = 10
Fs = 30
a, b = dyskretyzacja(f, Fs)
plt.plot(a, b)
plt.xlabel('t')
plt.ylabel('s(t)')
plt.title('f = 10[Hz], Fs = 30[Hz]')
plt.show()

#częstotliwość f = 10 Hz, próbkowanie Fs = 45 Hz
f = 10
Fs = 45
a, b = dyskretyzacja(f, Fs)
plt.plot(a, b)
plt.xlabel('t')
plt.ylabel('s(t)')
plt.title('f = 10[Hz], Fs = 45[Hz]')
plt.show()

#częstotliwość f = 10 Hz, próbkowanie Fs = 50 Hz
f = 10
Fs = 50
a, b = dyskretyzacja(f, Fs)
plt.plot(a, b)
plt.xlabel('t')
plt.ylabel('s(t)')
plt.title('f = 10[Hz], Fs = 50[Hz]')
plt.show()

#częstotliwość f = 10 Hz, próbkowanie Fs = 100 Hz
f = 10
Fs = 100
a, b = dyskretyzacja(f, Fs)
plt.plot(a, b)
plt.xlabel('t')
plt.ylabel('s(t)')
plt.title('f = 10[Hz], Fs = 100[Hz]')
plt.show()

#częstotliwość f = 10 Hz, próbkowanie Fs = 150 Hz
f = 10
Fs = 150
a, b = dyskretyzacja(f, Fs)
plt.plot(a, b)
plt.xlabel('t')
plt.ylabel('s(t)')
plt.title('f = 10[Hz], Fs = 150[Hz]')
plt.show()

#częstotliwość f = 10 Hz, próbkowanie Fs = 200 Hz
f = 10
Fs = 200
a, b = dyskretyzacja(f, Fs)
plt.plot(a, b)
plt.xlabel('t')
plt.ylabel('s(t)')
plt.title('f = 10[Hz], Fs = 200[Hz]')
plt.show()

#częstotliwość f = 10 Hz, próbkowanie Fs = 250 Hz
f = 10
Fs = 250
a, b = dyskretyzacja(f, Fs)
plt.plot(a, b)
plt.xlabel('t')
plt.ylabel('s(t)')
plt.title('f = 10[Hz], Fs = 250[Hz]')
plt.show()

#częstotliwość f = 10 Hz, próbkowanie Fs = 1000 Hz
f = 10
Fs = 1000
a, b = dyskretyzacja(f, Fs)
plt.plot(a, b)
plt.xlabel('t')
plt.ylabel('s(t)')
plt.title('f = 10[Hz], Fs = 100[Hz]')
plt.show()

#4
#Tak, istnieje takie twierdzenie.
#Nazywa się ono Twierdzenie o próbkowaniu, twierdzenie Shannona - Nyquista

#5
#Zjawisko to nazywa się aliasing – nieodwracalne zniekształcenie sygnału w procesie próbkowania wynikające z niespełnienia założeń twierdzenia o próbkowaniu.
#Zniekształcenie objawia się obecnością w wynikowym sygnale składowych o błędnych częstotliwościach (aliasów).

#7
# Aby wyświetlić obraz z aliasingiem używana jest interpolacja
# opcje: 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
# 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
# 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'


#KWANTYZACJA

img = mpimg.imread('obraz.png')
plt.imshow(img)
plt.show()

#2
print("Wymiary macierzy - wczytanego obrazka")
print(np.ndim(img))
print("------------")

#3
print("Wartości opisujące pojedynczy piksel")
print(img[60,45])
#Piksel opisywany jest przez 3 wartości
#print(img.shape[0]) szerokosc
#print(img.shape[1]) wysokosc
#print(img.shape[2]) glebokosc
print("------------")

#4
def jasnosc(rgb):
    return np.dot(rgb[...,:3],[(rgb.max()+rgb.min()/2),(rgb.max()+rgb.min()/2),(rgb.max()+rgb.min()/2)])

def usrednianie(rgb):
    nrgb = rgb.copy()
    nrgb[:,:,0] = (rgb[:,:,0] + rgb[:,:,1] + rgb[:,:,2])/3
    nrgb[:,:,1] = (rgb[:,:,0] + rgb[:,:,1] + rgb[:,:,2])/3
    nrgb[:,:,2] = (rgb[:,:,0] + rgb[:,:,1] + rgb[:,:,2])/3
    return nrgb

def luminacja(rgb):
    return np.dot(rgb[...,:3], [0.21, 0.72, 0.07])

#5
im1 = jasnosc(img)
im2 = usrednianie(img)
im3 = luminacja(img)

#np.histogram() - zwraca liczbę punktów w danym pojemniku

count1, b1 = np.histogram(im1, 255)
plt.imshow(im1, cmap=plt.get_cmap('gray'), vmin = 0, vmax = 1)
plt.title("Przekształcenie do skali szarości przez wyznaczanie jasności")
plt.show()
plt.hist(im1.flatten(), bins=255, range=(0.0, 1.0))
plt.title("Histogram 1.")
plt.show()

count2, b2 = np.histogram(im2, 255)
plt.imshow(im2, cmap=plt.get_cmap('gray'), vmin = 0, vmax = 1)
plt.title("Przekształcenie do skali szarości przez uśrednianie")
plt.show()
plt.hist(im2.flatten(), bins=256, range=(0.0, 1.0))
plt.title("Histogram 2.")
plt.show()


count3, b3 = np.histogram(im3, 255)
plt.imshow(im3, cmap=plt.get_cmap('gray'), vmin = 0, vmax = 1)
plt.title("Przekształcenie do skali szaości przez luminancję")
plt.show()
plt.hist(im3.flatten(), bins=255, range=(0.0, 1.0))
plt.title("Histogram 3.")
plt.show()

#6
count4, b4 = np.histogram(im1, 16)
print("Zakresy kolorów zredukowanych poprzez parametr 'bins'")
print(b4)
plt.hist(im1.flatten(), bins = b4, range=(0.0, 1.0))
plt.title("Histogram 4.")
plt.show()

#7
#Śreodek przedziału to pojedyncza wartość, zatem jeśli warość każdego piksela ustawię na wartość środka przedziału to otrzymam jednokolorowy obraz
def redukcja(rgb, s):
    nrgb = rgb.copy()
    nrgb[:,:,:] = rgb[s,s,s]
    return nrgb

count5, b5 = np.histogram(img, 16)
s = np.mean(b5)
s = s.astype('int')
im4 = redukcja(img, s)
plt.imshow(im4)
plt.title("Obraz o zredukowanej liczbie kolorów")
plt.show()

count6, b6 = np.histogram(im4, 16)
plt.hist(im4.flatten(), b6)
plt.title("Histogram 5.")
plt.show()


#BINARYZACJA
#2
img = mpimg.imread('obraz1.png')
plt.imshow(img)
plt.show()

im1 = usrednianie(img)
count1, b1 = np.histogram(im1, 255)
print(count1)
plt.imshow(im1, cmap=plt.get_cmap('gray'), vmin = 0, vmax = 1)
plt.title("Przekształcenie do skali szarości przez usrednianie")
plt.show()
plt.hist(im1.flatten(), bins=256, range=(0.0, 1.0))
plt.title("Histogram 6.")
plt.show()

#3
def binaryzacja(rgb):
    nrgb = rgb.copy()
    counter = 0
    pp = 0
    size = np.size(rgb)

    for i in range (rgb.shape[0]):
        for j in range (rgb.shape[1]):
            for k in range (rgb.shape[2]):
                counter = counter + rgb[i][j][k]

    pp = counter/size

    for i in range (rgb.shape[0]):
        for j in range (rgb.shape[1]):
            for k in range (rgb.shape[2]):
                if rgb[i][j][k] >= pp:
                    nrgb[i][j][k] = 255
                else:
                    nrgb[i][j][k] = 0
    return nrgb, pp

#4
im2, pp = binaryzacja(im1)
print("Punkt progowania: ")
print(pp)
plt.imshow(im2)
plt.title("Obraz zbinaryzowany")
plt.show()


