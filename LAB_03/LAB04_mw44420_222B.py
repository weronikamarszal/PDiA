import math;
# import Image
from PIL import Image
import matplotlib.image as mpimg
import numpy as np;
import matplotlib.pyplot as plt;
import cv2;


### 2. DYSKRETYZACJA ###

# ZAD1
def dyskretyzacja(czestotliwosc_sygnalu, czestotliwosc_probkowania):
    dt = 1 / czestotliwosc_probkowania;
    t = np.arange(0, 1, dt);
    s = [];
    i = 0
    for zmienne in t:
        s.append(math.sin(2 * math.pi * czestotliwosc_sygnalu * t[i]));
        i += 1;
    return s, t


# a)
s, t = dyskretyzacja(10, 20)
# print(s)
# print(t)

plt.plot(t, s)
plt.show()

s, t = dyskretyzacja(10, 21)
plt.plot(t, s)
plt.show()

# s, t = dyskretyzacja(10, 30)
# plt.plot(t, s)
# plt.show()
#
# s, t = dyskretyzacja(10, 45)
# plt.plot(t, s)
# plt.show()
#
# s, t = dyskretyzacja(10, 50)
# plt.plot(t, s)
# plt.show()
#
# s, t = dyskretyzacja(10, 100)
# plt.plot(t, s)
# plt.show()
#
# s, t = dyskretyzacja(10, 150)
# plt.plot(t, s)
# plt.show()
#
# s, t = dyskretyzacja(10, 200)
# plt.plot(t, s)
# plt.show()
#
# s, t = dyskretyzacja(10, 250)
# plt.plot(t, s)
# plt.show()
#
# s, t = dyskretyzacja(10, 1000)
# plt.plot(t, s)
# plt.show()

# ZAD4
# częstotliwość Nyquista - z jaką czesotliwoscia nalezy probkowac
# pełne odtworzenie sygnału ciągłego w czasie jest możliwe,
# jeśli częstotliwość próbkowania sygnału cyfrowego
# jest wyższa niż dwukrotność najwyższej częstotliwości sygnału próbkowanego

# zad5
# Aliasing - zjawisko, które z powodu błednie dobranej czestotliwosci
# próbkowania powoduje błedna interpretacje sygnału

# ZAD 6,7

# methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
#            'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
#            'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']
#
# np.random.seed(19680801)
#
# grid = np.random.rand(4, 4)
#
# fig, axs = plt.subplots(nrows=3, ncols=6, figsize=(9, 6),
#                         subplot_kw={'xticks': [], 'yticks': []})
#
# for ax, interp_method in zip(axs.flat, methods):
#     ax.imshow(grid, interpolation=interp_method, cmap='viridis')
#     ax.set_title(str(interp_method))
#
# plt.tight_layout()
# plt.show()


### 3. KWANTYZACJA ###

# ZAD1
im = Image.open("photo.png")
# im.show();

# ZAD2
data = np.asarray(im)
# print(len(data.shape))

# ZAD3
# print(im.ndim)

# ZAD4
## a) wyznaczanie jasnosci piksela
# gray1 = np.max(data, axis=2) + np.min(data, axis=2) / 2
# im = Image.fromarray(gray1)
# if im != 'RGB':
#     im = im.convert('RGB')
# im.save("gray1.png")
#
# # b) usrednianie wartosci piksela
# gray2 = (data[:, :, 0] + data[:, :, 1] + data[:, :, 2]) / 3
# im = Image.fromarray(gray2)
# if im != 'RGB':
#     im = im.convert('RGB')
# im.save("gray2.png")

## c) wyznaczanie luminacji piksela
# gray3 = (0.21 * data[:, :, 0] + 0.72 * data[:, :, 1] + 0.07 * data[:, :, 2])
# im = Image.fromarray(gray3)
# if im != 'RGB':
#     im = im.convert('RGB')
# x = im.save("gray3.png")


## ZAD.5
# hist1=np.histogram(gray1)
# plt.figure()
# plt.hist(gray1)
# plt.show()

# hist2=np.histogram(gray2)
# plt.figure()
# plt.hist(gray2)
# plt.show()
#
# hist3=np.histogram(gray3)
# plt.figure()
# plt.hist(gray3)
# plt.show()

# ZAD6
# x = np.histogram(gray2, bins=16);
# plt.figure();
# plt.hist(x)
# plt.show()

# ZAD7
#
# _, bins = np.histogram(im, 1)
# img_digitized = np.digitize(im, bins)
# new_img = bins[img_digitized - 1]
# plt.imshow(new_img)
# plt.show()

# ZAD8
# gray1zdj = Image.open("gray1.png")
# gray2zdj = Image.open("gray2.png")
# gray3zdj = Image.open("gray3.png")
# gray1zdj.show();
# gray2zdj.show();
# gray3zdj.show();


### 4. BINARYZACJA ###
# ZAD2
# zdj = Image.open("zdj.JPG")
# data2 = np.asarray(zdj)
# #
# zdj1 = (0.21*data2[:,:,0]+0.72*data2[:,:,1]+0.07*data2[:,:,2])
# zdj = Image.fromarray(zdj1)
# if zdj != 'RGB':
#     zdj = zdj.convert('RGB')
# x = zdj.save("zdj1.png")
# #
# histogram=np.histogram(zdj1)
# plt.figure()
# plt.hist(zdj1)
# plt.show()

def binaryzacja(zdj):
    zdj1 = zdj.copy()
    it = 0
    x = 0


    for i in range (zdj.shape[0]):
        for j in range (zdj.shape[1]):
            for k in range (zdj.shape[2]):
                it = it + zdj[i][j][k]
    x = it/np.size(zdj)
    for i in range (zdj.shape[0]):
        for j in range (zdj.shape[1]):
            for k in range (zdj.shape[2]):
                if zdj[i][j][k] >= x:
                    zdj1[i][j][k] = 255
                else:
                    zdj1[i][j][k] = 0
    return zdj1, x

im2, x = binaryzacja(im)
print(x)
plt.imshow(im2)
plt.show()
