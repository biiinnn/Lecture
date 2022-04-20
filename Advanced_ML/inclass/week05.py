# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 11:31:43 2021

@author: yebin
"""

from sklearn.svm import NuSVC
from sklearn.datasets import make_classification
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

X1, y1 =  make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, 
                              n_clusters_per_class=1, random_state=6)

plt.scatter(X1[:,0], X1[:,1], c=y1)

nusvc = NuSVC(kernel='linear', nu=0.1)
nusvc.fit(X1, y1)

nusvc.coef_
nusvc.intercept_
nusvc.dual_coef_
nusvc.n_support_
nusvc.support_

w = nusvc.coef_[0]
b = nusvc.intercept_[0]

xx = np.linspace(-3,4,100)
yy = -w[0]/w[1]*xx-b/w[1]

plt.scatter(X1[:,0], X1[:,1], c=y1)
plt.plot(xx,yy,'b')

X2, y2 = datasets.make_regression(n_samples=100, n_features=1, n_informative=1, noise=5)
plt.scatter(X2.flatten(), y2)

from sklearn.svm import SVR

svr = SVR(kernel='linear', epsilon=0.1)
svr.fit(X2, y2)

svr.coef_
svr.intercept_
svr.dual_coef_
svr.n_support_


b = 0.1
w = 1

x = np.random.rand(200)*2-1
y = w*x+b+np.random.randn(200)*0.2

svr.fit(x[:,None], y)

svr.dual_coef_
svr.n_support_

w = svr.coef_[0][0]
b = svr.intercept_[0]

xx = np.linspace(-1,1,100)
yy = w*xx+b

plt.scatter(x, y)
plt.plot(xx, yy, 'k')
plt.plot(xx, yy+0.3, ':r')
plt.plot(xx, yy-0.3, ':r')

svr.support_[np.abs(svr.dual_coef_[0])<1]

y_pred = svr.predict(x[:,None])
error = np.abs(y-y_pred)

svr.support_
error[1]

x = np.random.uniform(-4,4,100)
y = np.sin(x)+np.random.normal(size=100,scale=0.4)

plt.scatter(x, y)

svr2 = SVR(kernel='rbf', gamma=0.5, epsilon=0.3)
svr2.fit(x[:,None],y)

xx = np.linspace(-4,4,100)
yy = svr2.predict(xx[:,None])

plt.scatter(x,y)
plt.plot(xx, np.sin(xx), 'k', label='sin(x)')
plt.plot(xx,yy,'r:',label='SVR')
plt.legend(fontsize=14)

from sklearn.svm import NuSVR

svr3 = NuSVR(kernel='rbf', gamma = 1, nu = 0.1)
svr3.fit(x[:,None],y)

yy = svr3.predict(xx[:,None])

plt.scatter(x,y)
plt.plot(xx, np.sin(xx), 'k', label='sin(x)')
plt.plot(xx,yy,'r:',label='SVR')
plt.legend(fontsize=14)

svr2.score(x[:,None],y)
svr3.score(x[:,None],y)

X3, y3 = datasets.make_blobs(n_samples=300, n_features=2, centers=[[0,0],[1,5],[6,2]],
                           cluster_std=[1,1,2])
plt.scatter(X3[:,0], X3[:,1], c=y3)

from sklearn.svm import OneClassSVM

svdd = OneClassSVM(kernel='rbf', gamma=1, nu=0.3)
svdd.fit(X3)

l = svdd.predict(X3)

svdd.dual_coef_

xmin,xmax=X3[:,0].min()-0.5, X3[:,0].max()-0.5
ymin,ymax=X3[:,1].min()-0.5, X3[:,1].max()-0.5

X, Y = np.meshgrid(np.linspace(xmin,xmax,100), np.linspace(ymin,ymax,100))

Z=np.c_[X.ravel(),Y.ravel()]
Z_pred = svdd.decision_function(Z)

plt.contourf(X,Y,Z_pred.reshape(X.shape), levels=np.linspace(Z_pred.min(),0.5),
            cmap=plt.cm.Blues_r)
plt.contourf(X,Y,Z_pred.reshape(X.shape), levels=[0,Z_pred.max()], colors='orange')
plt.contour(X,Y,Z_pred.reshape(X.shape), levels=[0], linewidths=2, colors='k')
plt.scatter(X3[svdd.support_[svdd.dual_coef_[0]<1],0], X3[svdd.support_[svdd.dual_coef_[0]<1],1], marker='s',c='r')
plt.scatter(X3[svdd.support_[svdd.dual_coef_[0]==1],0], X3[svdd.support_[svdd.dual_coef_[0]==1],1], marker='x',c='k')
plt.scatter(X3[np.isin(np.arange(300), svdd.support_)==False,0], X3[np.isin(np.arange(300), svdd.support_)==False,1], marker='.',c='k')

















