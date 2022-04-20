# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 11:02:34 2021

@author: yebin
"""

from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier  # 사용할 알고리즘 import
import numpy as np

iris=datasets.load_iris()
X=iris.data
y=iris.target

bg_clf=BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3), n_estimators=5, oob_score=True)

bg_clf.fit(X,y)

bg_clf.base_estimator_
est=bg_clf.estimators_ #개별 classifier 포함
est[0].predict(X) 

len(np.unique(bg_clf.estimators_samples_[0]))
len(bg_clf.estimators_samples_[0])

bg_clf.estimators_features_
bg_clf.oob_score_

y_pred=bg_clf.predict(X)
y_prob=bg_clf.predict_proba(X)

from sklearn.tree import DecisionTreeClassifier

bg_clf=BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=5), n_estimators=100, oob_score=True)
bg_clf.fit(X[:,:2],y)

est=bg_clf.estimators_

xmin, xmax=X[:,0].min(),X[:,0].max()
ymin, ymax=X[:,1].min(),X[:,1].max()

XX,YY=np.meshgrid(np.linspace(xmin-0.5,xmax+0.5,100), np.linspace(ymin-0.5,ymax+0.5,100))
ZZ=np.c_[XX.ravel(), YY.ravel()]

ZZ_pred=bg_clf.predict(ZZ)

import matplotlib.pyplot as plt

plt.contourf(XX,YY,ZZ_pred.reshape(XX.shape), cmap=plt.cm.RdYlBu, alpha=0.7)
plt.scatter(X[:,0],X[:,1],s=10,c=y,cmap=plt.cm.RdYlBu)

ZZ_pred2=est[2].predict(ZZ)
plt.contourf(XX,YY,ZZ_pred2.reshape(XX.shape), cmap=plt.cm.RdYlBu, alpha=0.7)
plt.scatter(X[:,0],X[:,1],s=10,c=y,cmap=plt.cm.RdYlBu)

x=np.random.uniform(-4,4,100)
y=np.sin(x)+np.random.normal(size=100, scale=0.4)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

bg_reg=BaggingRegressor(base_estimator=KNeighborsRegressor(n_neighbors=3), oob_score=True)
bg_reg.fit(x.reshape((-1,1)),y)

bg_reg.oob_score_

est=bg_reg.estimators_

xx=np.linspace(-4,4,100)

for e in est:
    plt.plot(xx,e.predict(xx.reshape((-1,1))),color='gray', lw=0.5)
plt.plot(xx,bg_reg.predict(xx.reshape((-1,1))),color='r')
plt.scatter(x,y)

from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

X=iris.data
y=iris.target

ada_clf=AdaBoostClassifier()
ada_clf.fit(X,y)

ada_clf.estimator_weights_
ada_clf.estimator_errors_
y_pred_list=list(ada_clf.staged_predict(X))

list(ada_clf.staged_score(X,y))

boston=datasets.load_boston()
X=boston.data
y=boston.target

ada_reg=AdaBoostRegressor(loss='exponential')
ada_reg.fit(X,y)

list(ada_reg.staged_score(X,y))
ada_reg.estimator_weights_
ada_reg.estimator_errors_

from sklearn.ensemble import VotingClassifier

X=iris.data
y=iris.target

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

clf1=LogisticRegression(max_iter=300)
clf2=GaussianNB()
clf3=DecisionTreeClassifier(max_depth=5)

eclf=VotingClassifier(estimators=[('Logistic',clf1),('GNB',clf2),('DT',clf3)],voting='hard',weights=[0.5,0.3,0.2])
eclf.fit(X,y)
eclf.estimators_
eclf.predict(X)
eclf.score(X,y)

