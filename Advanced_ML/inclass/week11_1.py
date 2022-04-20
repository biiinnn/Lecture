# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 11:17:12 2021

@author: yebin
"""

from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np

X1,y1=datasets.make_moons(n_samples=200, noise=0.1, random_state=10)
flg=plt.figure(figsize=(8,6))
plt.scatter(X1[:,0],X1[:,1],c=y1)

X2,y2=datasets.make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=11)
flg=plt.figure(figsize=(6,6))
plt.scatter(X2[:,0],X2[:,1],c=y2)

dbscan=DBSCAN(eps=0.2, min_samples=3) #eps를 키우면 적은 집단으로 나눠짐 # min_samples를 키우면 많은 집단으로 나눠짐
dbscan.fit(X1)
cores=dbscan.core_sample_indices_ # core 샘플 확인
y1_l=dbscan.labels_ #음수(-1)는 outlier를 의미
dbscan.components_ #core point의 좌표 저장
plt.scatter(X1[y1_l!=-1,0],X1[y1_l!=-1,1],c=y1_l[y1_l!=-1])
plt.scatter(X1[y1_l==-1,0],X1[y1_l==-1,1], marker='x',c='k')

dbscan=DBSCAN(eps=0.15, min_samples=3) #eps(밀도느낌)를 키우면 적은 집단으로 나눠짐 # min_samples를 키우면 많은 집단으로 나눠짐
dbscan.fit(X2)
y2_1=dbscan.labels_ 
plt.scatter(X2[y2_1!=-1,0],X2[y2_1!=-1,1],c=y2_1[y2_1!=-1])
plt.scatter(X2[y2_1==-1,0],X2[y2_1==-1,1], marker='x',c='k')

import pandas as pd
#시카고 지역 범죄 발생 데이터
crime=pd.read_csv('https://drive.google.com/uc?export=download&id=1lwQ61ukX-iHagw-UFV1FvLW87tZw7aea')

crime_sel=crime.head(100)

crime['Primary Type'].value_counts()


from gmplot import gmplot

crime_type='ROBBERY'
gmap=gmplot.GoogleMapPlotter(41.832621, -87.658502, 11)
X=crime[crime['Primary Type']==crime_type][['Latitude','Longitude']].values
if len(X)>10000:
    ind=np.random.choice(range(len(X)), size=10000, replace=False)
else:
    ind=range(len(X))
gmap.scatter(X[ind,0],X[ind,1],size=40, marker=False)
gmap.heatmap(X[ind,0],X[ind,1])
gmap.draw(r'D:\%s.html'%(crime_type.replace(' ','_')))

unit=3280.84 #1km=3280.84feet

X2=crime[crime['Primary Type']==crime_type][['X Coordinate','Y Coordinate']].values
X2=X2[ind]
dbscan=DBSCAN(eps=0.5*unit, min_samples=50)
dbscan.fit(X2)

label=dbscan.labels_ #outlier가 존재
np.unique(label)

sel_ind=ind[label!=-1]

from matplotlib.colors import to_hex

colors=plt.cm.hsv(label[label!=-1]/label.max())
colors_hex=[to_hex(c) for c in colors] #RGB->HEXA

gmap=gmplot.GoogleMapPlotter(41.832621, -87.658502, 11)
gmap.scatter(X[sel_ind,0],X[sel_ind,1],size=40, marker=False, color=colors_hex)
gmap.heatmap(X[sel_ind,0],X[sel_ind,1])
gmap.draw(r'D:\%s.html'%(crime_type.replace(' ','_')))