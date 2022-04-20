# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 11:38:58 2021

@author: yebin
"""

### PCA

from sklearn.decomposition import PCA
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

X,y = datasets.make_regression(n_samples=1000, n_features=100, n_informative=3, effective_rank=10) # rank에 따라 linear 변동, 값을 모두 늘리면 linear하게 됨

pca=PCA()
pca.fit(X)

pca.mean_
X.mean(0)
comp=pca.components_  #W^T 와 동일
exp_var=pca.explained_variance_ #eigenvalue값과 동일


X0=X-pca.mean_  #평균을 0으로
X0.mean(0)

cov_mat=np.cov(X0.T)

np.diag(cov_mat).sum()

exp_var.sum()   #cov와 비슷, 거의 동일

t=np.dot(X0, comp[0]) #첫번째 주성분 projection
np.var(t)

exp_var[0] #거의 비슷

np.sum(pca.explained_variance_ratio_)

exp_var/exp_var.sum()

cum_exp_var_ratio=np.cumsum(pca.explained_variance_ratio_)

plt.plot(range(1,101),cum_exp_var_ratio)


T=pca.transform(X)

T2=np.matmul(X0, comp[:10].T)

T2.shape

plt.scatter(T2[:,0], T2[:,1])


### MDS

from sklearn.manifold import MDS
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

iris=datasets.load_iris()
X=iris.data
y=iris.target

mds=MDS(n_components=2)
pca=PCA(n_components=2)

pca.fit(X)
X_pca=pca.transform(X)

mds.fit(X)
X_mds=mds.fit_transform(X)

plt.scatter(X_pca[:,0],X_pca[:,1],c=y)

plt.scatter(X_mds[:,0],X_mds[:,1],c=y) #학습할때마다 결과값 달라짐

eurodist=pd.read_csv('https://drive.google.com/uc?export=download&id=1fXiYlm22PhsrR4RnEcIhfQfhH1-KMvz7', index_col=0)

mds=MDS(n_components=2, dissimilarity='precomputed')
X_euro=mds.fit_transform(eurodist)

fig=plt.figure(figsize=(12,12))
plt.scatter(X_euro[:,0],X_euro[:,1])

for country, (x,y) in zip(eurodist.index, X_euro):
    plt.text(x,y,country,fontsize=14)



### feature selection

from sklearn.feature_selection import chi2, mutual_info_classif, f_classif
import numpy as np
X=iris.data
y=iris.target

chi2(X,y) # 첫번째 줄: 원본 값, 두번째 줄: p-value 값 측정

h1=np.histogram(X[:,0],bins=5) #categorization

X[:,[0]]>h1[1][1:]

Xcat=np.zeros(X.shape, dtype=int)
for i in range(4):
    h1=np.histogram(X[:,i],bins=5)
    Xcat[:,i]=np.sum(X[:,[i]]>h1[1][1:],axis=1)

chi2(Xcat,y)

f_classif(X,y) #oneway ANOVA


### filter

mutual_info_classif(Xcat, y, discrete_features=True)

from sklearn.feature_selection import f_regression, mutual_info_regression

boston=datasets.load_boston()
Xb=boston.data
yb=boston.target

f_regression(Xb,yb)

mutual_info_regression(Xb, yb, discrete_features=False)

sms=pd.read_csv('https://drive.google.com/uc?export=download&id=1l6gUFvs4PNoY2OVg44hCNmOREfEsx2qX')

sms.max()

bi_sms=(sms>0)*1

chi2_result=chi2(bi_sms.drop('target',axis=1),bi_sms['target'])

order=np.argsort(chi2_result[0])[::-1]

bi_sms.loc[order[:100]]
bi_sms.columns[order[:100]]


from sklearn.feature_selection import SelectKBest

X_sms=bi_sms.drop('target',axis=1)
y_sms=bi_sms['target']

fs=SelectKBest(chi2, k=10)
fs.fit(X_sms, y_sms)

fs.scores_

Xred=fs.transform(X_sms)
