# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 13:15:51 2021

@author: yebin
"""
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

X1 , y1 = datasets.make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=6)

plt.scatter(X1[:,0], X1[:,1], c=y1, alpha=0.5)

clf1 = SVC(kernel='linear', C=100)
clf1.fit(X1, y1)

clf1.support_
clf1.dual_coef_
clf1.support_vectors_
clf1.n_support_
clf1.coef_
clf1.intercept_


w = clf1.coef_[0]
xx = np.linspace(-3,4,100)
yy = -w[0]/w[1]*xx-clf1.intercept_/w[1]

plt.scatter(X1[:,0], X1[:,1], c=y1, alpha=0.5)
plt.plot(xx,yy)

from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression()
logistic.fit(X1, y1)

logistic.coef_
logistic.intercept_

w2 = logistic.coef_[0]
yy2 = -w2[0]/w2[1]*xx-logistic.intercept_/w2[1]


plt.scatter(X1[:,0], X1[:,1], c=y1, alpha=0.5)
plt.plot(xx,yy2)

X2, y2 = datasets.make_moons(n_samples=200, noise=0.17, random_state=33)

plt.scatter(X2[:,0], X2[:,1], c=y2)

clf2 = SVC(kernel='poly', degree=10, C=1)
clf2.fit(X2, y2)

clf2.support_
clf2.n_support_
clf2.dual_coef_
#clf2.coef_
clf2.intercept_

X, Y = np.meshgrid(np.linspace(-2,3,100), np.linspace(-1,2,100))

Z = np.c_[X.ravel(), Y.ravel()]
Z_pred = clf2.predict(Z)

plt.scatter(X2[:,0], X2[:,1], c=y2)
plt.contourf(X, Y, np.reshape(Z_pred, X.shape), alpha=0.5)

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

tr = DecisionTreeClassifier(max_depth=3)
tr.fit(X2, y2)

Z_pred_tr = tr.predict(Z)

plt.scatter(X2[:,0], X2[:,1], c=y2)
plt.contourf(X, Y, np.reshape(Z_pred_tr, X.shape), alpha=0.5)

nb = GaussianNB()
nb.fit(X2, y2)

Z_pred_nb = nb.predict(Z)
plt.scatter(X2[:,0], X2[:,1], c=y2)
plt.contourf(X, Y, np.reshape(Z_pred_nb, X.shape), alpha=0.5)




