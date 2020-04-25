import numpy as np
from numpy.lib.stride_tricks import as_strided
import random

# a = np.array([1, 2, 3, 4, 5, 6, 7]);
# print(a);

# b=np.array([[1,2,3,4,5], [6,7,8,9,10]]);
# print(b);

# c=np.transpose(b);
# print(c);

# d=np.arange(100);
# print(d);

# e=np.linspace(1,2,10);
# print(e);

# f=np.arange(0,101,5);
# print(f);

# g=np.random.rand(20).round(2);
# print(g);

# h=np.random.randint(1,1000,100);
# print(h);

# i=np.zeros((3,2));
# print(i);

# j=np.ones((3,2));
# print(j);

# k=np.random.randint(10000,size=(5,5));
# k=np.float32(k);
# print(k);

# liczby losowe - zad
# a=np.random.rand(10)*10 #mnozenie przez 10 zeby byly calkowite
# b=a.astype(int) #zamiana wartosci na int
# a=np.round(a,0) #0-tyle liczb po przecinku
# a=a.astype(int)
# print(a)
# print()
# print(b)

###SELEKCJA DANYCH
# b=np.array([[1,2,3,4,5], [6,7,8,9,10]],dtype=np.int32)
# print("ile wymiarow: ", b.ndim); #sprawdzenie ile b ma wymiarow - 2
# print("ile elementow: ", b.size);

#wybranie wartosci 2 i 4 z tablicy
# tab=[]
# for i in b:
#     for j in i:
#         if j==2 or j==4:
#             tab.append(j)
# print(tab)

# print(b[0,:]) #pierwszy wiersz
# print(b[:,0]) #pierwsza kolumna

# x=np.random.randint(0,100,(20,7)) #macierz 20x7, przedzial 0-100
# print(x[:,0:4]) # 4 pierwsze kolumny

### OPERACJE MATEMATYCZNE I LOGICZNE
# a=np.random.randint(0,10,(3,3))
# b=np.random.randint(0,10,(3,3))

# c=np.add(a,b); #dodawnie macierzy
# d=np.np.multiply(a,b); #mnozenie macierzy
# e=e=np.multiply(a,np.linalg.inv(b)) #dzielenie macierzy
# f=a^b #potegowanie macierzy

# print(a>=4);

# print(b.diagonal().sum()) #suma elementow na glownej przekatnej

### DANE STATYSTYCZNE

#suma elementow macierzy
# sum=0
# for i in b:
#     for j in i:
#          sum=sum+j
# print(sum)

# print(np.min(b)) #wart min
# print(np.max(b))# wart max
# print(np.std(b)) # odchylenie std
# print(np.mean(b[1,:])) # srednia dla pierwszego wiersza
# print(np.mean(b[:,2])) # srdnia dla drugiej kolumny

### RZUTOWANIE WYMIARÓW ZA POMOCA RESHAPE LUB RESIZE

# a=np.arange(0,50,1)
# b=np.reshape(a, (10,5)) # reshape - tworzy z tablocy macierz o podanych wymiarach
# c=np.resize(a, (10,5)) #resize - tworzy z tablicy macierz o podanych wymiarach
# print(a)
# print()
# print(b)
# print()
# print(c)

#Komenda ravel zwraca ciągłą spłaszczoną tablicę

# d=np.array([1,2,3,4,5]);
# e=np.array([1,2,3,4]);
# #Komenda newaxis służy do tego aby zwiększyć wymiar tablicy o 1
# e=e[:,np.newaxis]
# print(d+e);

### SORTOWANIE DANYCH
# a=np.random.rand(5,5);
# print(np.sort(a,axis=1)) #sortowanie wierszy rozsnaco
# print(np.sort(a,axis=0)[::-1]) #sortowanie kulumn malejaco

# b=np.array([(1, 'MZ', 'mazowieckie'),
#             (2, 'ZP', 'zachodniopomorskie'),
#             (3, 'ML', 'małopolskie')]);
#
# c=np.resize(b, (3,3));
# d=b[b[:,1].argsort()] #sortowanie rosnaco po kolumnie 2
# print(d[2,2])



#########################
# ZADANIA PODSUMOWUJACE

#ZAD1
# A = np.random.randint(100, size=(10, 5))
# print(A);
# print(np.trace(A));
# print(np.diag(A));

#ZAD2
# tab1=np.random.normal(size=10)
# tab2=np.random.normal(size=10)
# tab3=tab1*tab2;
# print(tab1,tab2,tab3);

#ZAD3
# tab1=np.random.randint(1,100,size=(1,5));
# tab2=np.random.randint(1,100,size=(1,5));
#
# A = tab1+tab2;
# print(A);

#ZAD4+ZAD5
# A=np.random.randint(100, size=(4,5));
# B=np.random.randint(100, size=(5,4));
# C = A+np.transpose(B);
# # print(C);
#
# D=A[:,3]*C[:,4];
# print(D);

#ZAD6
# print("normal");
# A=np.random.normal(size=(3,3));
# print(A.mean(), A.std(), A.var());
# B=np.random.normal(size=(3,3));
# print(B.mean(), B.std(), B.var());
# print("uniform");
# C=np.random.normal(size=(3,3));
# print(C.mean(), C.std(), C.var());
# D=np.random.normal(size=(3,3));
# print(D.mean(), D.std(), D.var());

#ZAD7
# A=np.random.randint(50,size=(4,4))
# B = np.random.randint(50,size=(4,4))
#
# C=A*B;
# print(C);
#
# D=np.dot(A,B); #iloczyn skalarny
# print(D);

#ZAD8
# A=np.arange(24).reshape(4,6);
# print(A);
# print(as_strided(A[0][2:], (3,5), A.strides));

#ZAD.9
# A = np.random.randint(10,size=3);
# B = np.random.randint(10,size=3);
# C = np.vstack((A,B));
# D = np.hstack((A,B));
#
# print(C, D);
#vstack dodaje kolejny wiersz i tworzy macierz dwuwymiarową,
# a hstack tworzy tablicę jednowymiarowa

#ZAD10
# A=np.arange(24).reshape(4,6);
# print(A);
# yellow = as_strided(A[0][0:3], (2,3), A.strides);
# print(yellow)
# green = as_strided(A[0][3:], (2,3), A.strides);
# blue = as_strided(A[2][0:3], (2,3), A.strides);
# orange = as_strided(A[2][3:], (2,3), A.strides);
# print(yellow.max());
# print(green.max());
# print(blue.max());
# print(orange.max());
