import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import scipy
from scipy import stats


#MANIPULOWANIE DANYMI
# dates = pd.date_range('20200301', periods=5)
# df = pd.DataFrame(np.random.normal(size=(5,3)), index=dates);
# df.index.names = ['data']
# df.columns = ['A', 'B', 'C'];
# print(df)

# id=np.arange(0,20,1)
# df=pd.DataFrame(np.random.randint(100000, size=(20,3)), index=id, columns=list('ABC'));
# df.index.names = ['id']
# print(df);
#
# print(df.head(3))
# print(df.tail(3))
# print(df.index.names)
# print(df.columns)
# print(df.to_string(index=False, header=False))
# print(df.sample(5))
# print(df.to_string(index=False, header=False))
# print(df.loc[:, ('A')].to_string(index=False));
# print(df.loc[:, ('A','B')].to_string(index=False));

# print(df.iloc[:3, :2]);
# print(df.iloc[4, :]);
# print(df.iloc[[0,5,6,7], [1,2]]);

###STATYSTYKI TABELI
# print(df.describe())
# print(df.ge(0))
# print(df.mean())


### laczenie tabel
# a = pd.DataFrame([['a' , 'b'], ['c', 'd']])
# b = pd.DataFrame([[1,2], [3,4]])
#
# x=pd.concat([a, b])
# print(x)
# print(x.T) #transpozycja

###SORTOWANIE
# df = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': ['a', 'b', 'a', 'b', 'b']})
# print(df)
# print(df.sort_index(ascending=False)); #sortowanie malejaco


# slownik = {'Day': ['Mon', 'Tue', 'Mon', 'Tue', 'Mon'], 'Fruit': ['Apple',
# 'Apple', 'Banana', 'Banana', 'Apple'], 'Pound': [10, 15, 50, 40, 5], 'Profit':[20, 30, 25, 20, 10]}
# df3 = pd.DataFrame(slownik)
# print(df3)
# print(df3.groupby('Day').sum())
# print(df3.groupby(['Day','Fruit']).sum())

# df=pd.DataFrame(np.random.randn(20, 3), index=np.arange(20), columns=['A','B','C'])
# df.index.name='id'
# print(df)
# df['B']=1 #cala kolumna 'B' wypelniona jedynkami
# print(df)
# df.iloc[1,2]=10 #wiersz 1, kolumna 2 = 10 (numerujemy od 0)
# print(df)
# df[df¡0]=-df #?????????
# print(df)


# df=pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': ['a', 'b', 'a', 'b', 'b']})
# s=pd.Series({'x': [1, 2, 3, 4, 5], 'y': ['a', 'b', 'a', 'b', 'b']})




# ###Zad1###
# x= df.groupby('x').mean();
# print(x)
#
# ###Zad2###
# # print(df.y.value_counts())
# # print(df.x.value_counts())
#
# ###ZAD3###
data = pd.read_csv('autos.csv');
# # print(data);
#
# data2 = np.loadtxt('autos.csv', dtype='str', skiprows=1, delimiter=',');
# # print(data2);
#
# #loadtxt wymaga definiowania typu danych, readcsv sam się domyśla
#
# ###ZAD4###
# zuzycie_paliwa=(data.groupby('make').mean().loc[:,'wheel-base'])
# # print(zuzycie_paliwa)
#
# ###ZAD5###
# licznosci=data.groupby(['make','fuel-type'])['fuel-type'].value_counts()
# # print(licznosci);
#
# ###ZAD6###
# pierwszy_stopien = np.polyfit(data["city-mpg"],data["length"],deg=1)
# drugi_stopien = np.polyfit(data["city-mpg"],data["length"],deg=2)
#
# print(pierwszy_stopien, drugi_stopien)

# #
# ###ZAD7###
# print(scipy.stats.pearsonr(data['length'], data['city-mpg']))
#
# ###ZAD8###
# plt.plot(data['length'], data['city-mpg'],'.')
# plt.plot(np.unique(data['length']), np.poly1d(np.polyfit(data['length'], data['city-mpg'], 1))(np.unique(data['length'])))
# plt.show()

#
# ###ZAD9###
estymator=stats.gaussian_kde(data['length'])
#
# plt.figure();
# plt.plot(data['length'], estymator(data['length']), 'g', label = 'Wykres')
# plt.plot(data['length'], estymator(data['length']), '.r', label = 'Próbki')
# plt.legend(loc='best')
# plt.show()
#
#
# ###ZAD10###
# plt.figure()
# ax=plt.subplot(2,1,1)
#
# plt.plot(data['length'], estymator(data['length']), 'g', label = 'Wykres')
# plt.plot(data['length'], estymator(data['length']), '.r', label = 'Próbki')
# plt.legend(loc='best')
# plt.ylabel('Oś Y')
# plt.xlabel('Oś X')
#
#
# ax=plt.subplot(2,1,2)
# estymator=stats.gaussian_kde(data['width'])
# plt.plot(data['width'], estymator(data['width']), 'r', label = 'Wykres')
# plt.plot(data['width'], estymator(data['width']), '.g', label = 'Próbki')
# plt.legend(loc='best')
# plt.ylabel('Oś Y')
# plt.xlabel('Oś X')
#
# plt.show()

###ZAD11###
# val = np.vstack([data['width'],data['length']])
# ker = scipy.stats.gaussian_kde(val)
#
# xmin = data['width'].min()
# xmax = data['width'].max()
# ymin = data['length'].min()
# ymax = data['length'].max()
# xx, yy = np.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100))
# pos = np.vstack([xx.ravel(),yy.ravel()])
# f = np.reshape(ker(pos).T, xx.shape)
#
# fig = plt.figure()
# ax = fig.gca()
# ax.set_xlim(xmin, xmax)
# ax.set_ylim(ymin, ymax)
#
# plt.plot(data['width'], data['length'], 'g.')
# cfset = ax.contour(xx, yy, f, cmap='Blues')
# cset = ax.contour(xx, yy, f, colors = 'g')
# ax.clabel(cset, inline=1, fontsize=12)
# ax.set_xlabel('Oś X')
# ax.set_ylabel('Oś Y')
#
# plt.savefig('wynik.png')
# plt.savefig('wynik.pdf')
# plt.show()




