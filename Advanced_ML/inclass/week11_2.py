# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 11:47:23 2021

@author: yebin
"""

import numpy as np
import matplotlib.pyplot as plt

x1=np.random.multivariate_normal([0,0], [[1,0.5],[0.5,1]],100)
x2=np.random.multivariate_normal([4,4], [[0.3,-0.1],[-0.1,0.3]],100)
x=np.concatenate((x1,x2),axis=0)
y=[0]*100+[1]*100

plt.scatter(x[:,0],x[:,1],c=y, alpha=0.1)

from sklearn.mixture import GaussianMixture

gmm=GaussianMixture(n_components=2, covariance_type='full')
gmm.fit(x)

gmm.weights_ #각각의 가우시안에 대한 사전확률, 파이 값
gmm.means_ #평균 벡터
gmm.covariances_ #공분산 행렬

cls_prob=gmm.predict_proba(x) #responsability 값

label=gmm.predict(x) #cls_prob 중 큰 값

log_prob=gmm.score(x) #전체 data에 대한 값

plt.scatter(x[:,0],x[:,1],c=label, alpha=0.7)

X,Y=np.meshgrid(np.arange(x[:,0].min()-5, x[:,0].max()+5,0.1),np.arange(x[:,1].min()-5, x[:,1].max()+5,0.1))
Z=np.c_[X.ravel(), Y.ravel()]
Z_prob=-gmm.score_samples(Z) #log probility 값이라 음수도 나옴

from matplotlib.colors import LogNorm

CS=plt.contour(X,Y,Z_prob.reshape(X.shape), norm=LogNorm(vmin=1, vmax=120), levels=np.logspace(0,2,20))
CB=plt.colorbar(CS, shrink=1)
plt.scatter(x[:,0],x[:,1])

