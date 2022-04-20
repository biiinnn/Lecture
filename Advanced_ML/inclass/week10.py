# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 10:35:42 2021

@author: yebin
"""

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

# data1
n=500
x1,y1=datasets.make_blobs(n_samples=n, random_state=8)
plt.scatter(x1[:,0],x1[:,1],c=y1)
# data2
x2,y2=datasets.make_blobs(n_samples=n, random_state=170)
plt.scatter(x2[:,0],x2[:,1],c=y2)
transformation=[[0.6,-0.6],[-0.4,0.8]]
x2=np.dot(x2, transformation)
plt.scatter(x2[:,0],x2[:,1],c=y2)
# data3
x3,y3=datasets.make_blobs(n_samples=n, cluster_std=[1.0,2.0,0.5], random_state=32)
plt.scatter(x3[:,0],x3[:,1],c=y3)
# data4
x4,y4=datasets.make_moons(n_samples=n, noise=0.05) #noise가 작아질수록 선에 가까워짐
plt.scatter(x4[:,0],x4[:,1],c=y4)

# clustering
from sklearn.cluster import KMeans, AgglomerativeClustering
#k-means
# data1
kmeans=KMeans(n_clusters=3)
kmeans.fit(x1)
kmeans_label=kmeans.labels_
plt.scatter(x1[:,0],x1[:,1],c=kmeans_label) # 할 때마다 색깔 달라짐
# data2
kmeans=KMeans(n_clusters=3)
kmeans.fit(x2)
kmeans_label=kmeans.labels_
plt.scatter(x2[:,0],x2[:,1],c=kmeans_label) # kmeans는 유클리디안 기준으로 정확하게 모양을 맞게 반영하지 못함
# data3
kmeans=KMeans(n_clusters=3)
kmeans.fit(x3)
kmeans_label=kmeans.labels_
plt.scatter(x3[:,0],x3[:,1],c=kmeans_label) # 약간 붙어있어서 정확하게 잘 나누지 못함
##
centroid=kmeans.cluster_centers_
centroid
## 새로운 data3
x3_new,y3_new=datasets.make_blobs(n_samples=n, cluster_std=[1.0,2.0,0.5], random_state=11)
plt.scatter(x3_new[:,0],x3_new[:,1],c=y3_new)

kmeans_label_new=kmeans.predict(x3_new)

plt.scatter(x3[:,0],x3[:,1],c=kmeans_label)
plt.scatter(centroid[:,0],centroid[:,1],c='k', marker='x')

plt.scatter(x3_new[:,0],x3_new[:,1],c=kmeans_label_new)
plt.scatter(centroid[:,0],centroid[:,1],c='k', marker='x')
#data4
kmeans=KMeans(n_clusters=2)
kmeans.fit(x4)
kmeans_label=kmeans.labels_
centroid=kmeans.cluster_centers_

plt.scatter(x4[:,0],x4[:,1],c=kmeans_label)
plt.scatter(centroid[:,0],centroid[:,1],c='k', marker='x')

#Agglomerative
# data1
aggl=AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='single')
aggl.fit(x1)
aggl_label=aggl.labels_
plt.scatter(x1[:,0],x1[:,1],c=aggl_label)
# data2
aggl=AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='complete')
aggl.fit(x2)
aggl_label=aggl.labels_
plt.scatter(x2[:,0],x2[:,1],c=aggl_label)

aggl=AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
aggl.fit(x2)
aggl_label=aggl.labels_
plt.scatter(x2[:,0],x2[:,1],c=aggl_label)
# data3
aggl=AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
aggl.fit(x3)
aggl_label=aggl.labels_
plt.scatter(x3[:,0],x3[:,1],c=aggl_label)

aggl=AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='single')
aggl.fit(x3)
aggl_label=aggl.labels_
plt.scatter(x3[:,0],x3[:,1],c=aggl_label)

aggl=AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='complete')
aggl.fit(x3)
aggl_label=aggl.labels_
plt.scatter(x3[:,0],x3[:,1],c=aggl_label)
# data4
aggl=AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')
aggl.fit(x4)
aggl_label=aggl.labels_
plt.scatter(x4[:,0],x4[:,1],c=aggl_label)

aggl=AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
aggl.fit(x4)
aggl_label=aggl.labels_
plt.scatter(x4[:,0],x4[:,1],c=aggl_label)

# 어떻게 묶이는지
x=[[30,10],[26,11],[16,16],[20,17],[19,18]]

aggl=AgglomerativeClustering(n_clusters=1, affinity='euclidean', linkage='ward')
aggl.fit(x)

aggl.children_ #묶이는 순서 확인 가능

# 덴드로그램 그리기
from scipy.cluster import hierarchy

Z=hierarchy.linkage(x1, method='single', metric='euclidean')
hierarchy.dendrogram(Z)

cls_label=hierarchy.cut_tree(Z, n_clusters=3)
cls_label.shape

cls_label=hierarchy.cut_tree(Z, n_clusters=[3,5,7])
cls_label.shape

cls_label=hierarchy.cut_tree(Z, height=2)

# 성능평가
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score, adjusted_rand_score

iris=datasets.load_iris()
X=iris.data
y=iris.target

n_cls=2
kmeans=KMeans(n_clusters=n_cls)
aggl=AgglomerativeClustering(n_clusters=n_cls)

kmeans.fit(X)
aggl.fit(X)

kmeans_label=kmeans.labels_
aggl_label=aggl.labels_

silhouette_score(X, kmeans_label) # 1에 가까울수록 좋음
homogeneity_score(y, kmeans_label) # ground truth와 실제 label 결과 넣어줘야 함
completeness_score(y, kmeans_label)
adjusted_rand_score(y, kmeans_label)

silhouette_score(X, aggl_label) # aggl_label의 지표가 더 좋게 나왔음
homogeneity_score(y, aggl_label)
completeness_score(y, aggl_label)
adjusted_rand_score(y, aggl_label)

kmeans.inertia_
inertia=[]
for n_cls in range(2,11):
    kmeans.n_clusters=n_cls
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
    
plt.plot(range(2,11), inertia)    # elbow 확인

from sklearn.metrics import silhouette_samples #개별 확인

kmeans=KMeans(n_clusters=3)
kmeans.fit(x1)
kmeans_label=kmeans.labels_

s = silhouette_samples(x1, kmeans_label)
count=0
fig=plt.figure(figsize=(6,8))
ax=plt.gca()
colors=plt.cm.hsv(np.arange(3)/3)
ylabel_pos=[]
for i in range(3):
    ind=np.where(kmeans_label==i)[0]
    sel_s=s[ind]
    sel_s.sort()
    ax.fill_betweenx(np.arange(count, count+len(ind)), 0, sel_s, fc=colors[i], ec=colors[i],alpha=0.7)
    ylabel_pos.append(count+len(ind)/2)
    count+=len(ind)
plt.yticks(ylabel_pos, [str(i) for i in range(3)]) # 위치 지정
plt.ylabel('Cluster label')
plt.xlabel('The silhoutte cofficient values')
ax.axvline(x=np.mean(s), color='r', ls='--') # 실루엣 분포 확인 가능

# data 바꿔서 보기
kmeans=KMeans(n_clusters=3)
kmeans.fit(x2)
kmeans_label=kmeans.labels_

s= silhouette_samples(x2, kmeans_label)
count=0
fig=plt.figure(figsize=(6,8))
ax=plt.gca()
colors=plt.cm.hsv(np.arange(3)/3)
ylabel_pos=[]
for i in range(3):
    ind=np.where(kmeans_label==i)[0]
    sel_s=s[ind]
    sel_s.sort()
    ax.fill_betweenx(np.arange(count, count+len(ind)), 0, sel_s, fc=colors[i], ec=colors[i],alpha=0.7)
    ylabel_pos.append(count+len(ind)/2)
    count+=len(ind)
plt.yticks(ylabel_pos, [str(i) for i in range(3)])
plt.ylabel('Cluster label')
plt.xlabel('The silhoutte cofficient values')
ax.axvline(x=np.mean(s), color='r', ls='--') # 실루엣 분포 확인 가능
