# -*- coding: utf-8 -*-
# DO NOT CHANGE
import numpy as np
from itertools import product
from sklearn.svm import OneClassSVM
from scipy.sparse.csgraph import connected_components
import pandas as pd
import matplotlib.pyplot as plt

def get_adj_mat(X,svdd,num_cut):
    # X: n*p input matrix
    # svdd: trained svdd model by sci-kit learn using X
    # num_cut: number of cutting points on line segment
    #######OUTPUT########
    # return adjacent matrix size of n*n (if two points are connected A_ij=1)
    
    # model fitting
    svdd.fit(X)
    
    # bounded support vector 경계 밖 위치
    bsv = svdd.support_[svdd.dual_coef_[0] == 1]
    # unbounded support vector 경계에 위치
    ubsv = svdd.support_[svdd.dual_coef_[0] != 1]
    # non support vector 경계 안 위치
    nonsv = np.setdiff1d(np.arange(X.shape[0]),np.append(bsv,ubsv))

    # 경계 포함 안에 있는 경우 조합
    pairs = list(product(ubsv, nonsv))
    
    # make zero matrix: 0으로 이루어진 행렬 생성, 행렬의 크기 = input data의 행 수 * input data의 행 수
    adj = np.zeros((X.shape[0],X.shape[0]))
    
    # get adjacency matrix
    for pair in pairs:
        # 경계 포함 안에 있는 경우 조합의 첫번째 점
        x1, y1 = X[pair[0]] 
        # 경계 포함 안에 있는 경우 조합의 두번째 점
        x2, y2 = X[pair[1]] 
        # 두 점 사이를 이어 num_cut의 수 만큼 나누어서 체크
        check_point = np.c_[np.linspace(x1, x2, num_cut), np.linspace(y1, y2, num_cut)]
        # check_point(각 점의 위치 제외)에서 예측 값(1또는 -1)의 합이 num_cut-2(각 점의 위치 제외)이면 같은 클러스터에 위치
        if sum(svdd.predict(check_point[1:-1])) == num_cut-2: 
            adj[pair[0]][pair[1]] = 1 
            adj[pair[1]][pair[0]] = 1
        
    return adj
    
def cluster_label(A,bsv):
    # A: adjacent matrix size of n*n (if two points are connected A_ij=1)
    # bsv: index of bounded support vectors
    #######OUTPUT########
    # return cluster labels (if samples are bounded support vectors, label=-1)
    # cluster number starts from 0 and ends to the number of clusters-1 (0, 1, ..., C-1)
    # Hint: use scipy.sparse.csgraph.connected_components
    
    # bounded support vector 경계 밖 위치
    bsv = svdd.support_[svdd.dual_coef_[0] == 1]
    
    # 인접행렬에서 connected_components
    n_components, labels = connected_components(A)
    
    # 경계 밖에 위치한 경우 label은 -1
    labels[bsv] = -1 # bounded support vectors, label=-1
    
    # 중복제외 labels 집합에서 bounded sv에 해당된 라벨(-1)값 빼기 => 고유한 라벨 list
    unique_label = list(set(labels)-{-1})
    
    # 클러스터 수(n_components)로 라벨링 다시 수행 0,1,2,...,(n_components-1)
    labels_dict = {-1:-1}
    
    i = 0
    for label in unique_label:
        labels_dict[label] = i
        i += 1
    
    new_labels = []
    for old in labels:
        new = labels_dict[old]
        new_labels.append(new)
    new_labels = np.array(new_labels)
    
    return new_labels


ring=pd.read_csv('https://drive.google.com/uc?export=download&id=1_ygiOJ-xEPVSIvj3OzYrXtYc0Gw_Wa3a')
num_cut=20
svdd=OneClassSVM(gamma=1, nu=0.2)

# input matrix
X = ring.to_numpy()
# adjacency matrix
A = get_adj_mat(X,svdd,num_cut)

# bounded support vector 경계 밖 위치
bsv = svdd.support_[svdd.dual_coef_[0] == 1]
# unbounded support vector 경계에 위치
ubsv = svdd.support_[svdd.dual_coef_[0] != 1]
# non support vector 경계 안 위치
nonsv = np.setdiff1d(np.arange(X.shape[0]),np.append(bsv,ubsv))

# cluster labels
label = cluster_label(A,bsv)

##########Plot1###################
# Get SVG figure (draw line between two connected points with scatter plots)
# draw decision boundary
# mark differently for nsv, bsv, and free sv

# make plot 격자 생성
xmin,xmax = ring.iloc[:,0].min()-0.5,ring.iloc[:,0].max()+0.5
ymin,ymax = ring.iloc[:,1].min()-0.5,ring.iloc[:,1].max()+0.5

XX, YY = np.meshgrid(np.linspace(xmin,xmax,100), np.linspace(ymin,ymax,100))
Z = np.c_[XX.ravel(),YY.ravel()]
Z_pred = svdd.decision_function(Z)

# draw decision boundary 경계 표시
plt.figure(figsize=(10,8))
plt.contour(XX,YY,Z_pred.reshape(XX.shape), levels=[0], linewidths=2, colors='k')
# mark points 각 점 표시 
plt.scatter(ring.iloc[bsv,0], ring.iloc[bsv,1], marker='x', color='blue')
plt.scatter(ring.iloc[ubsv,0], ring.iloc[ubsv,1], marker='o', facecolor='none', color='red')
plt.scatter(ring.iloc[nonsv,0], ring.iloc[nonsv,1], marker='o', color='k')
# draw line 경계 포함 안에 위치한 경우 점과 점을 이어주는 선 그리기
for i in range(A.shape[0]):
    for j in range(i,A.shape[0]):
        if A[i][j] == 1:
            plt.plot([ring.iloc[:, 0][i],ring.iloc[:, 0][j]], [ring.iloc[:, 1][i],ring.iloc[:, 1][j]] , 'k')

##########Plot2###################
# Clsuter labeling result
# different clusters should be colored using different color
# outliers (bounded support vectors) are marked with 'x'

# make plot 격자 생성
xmin,xmax = ring.iloc[:,0].min()-0.5,ring.iloc[:,0].max()+0.5
ymin,ymax = ring.iloc[:,1].min()-0.5,ring.iloc[:,1].max()+0.5

XX, YY = np.meshgrid(np.linspace(xmin,xmax,100), np.linspace(ymin,ymax,100))
Z = np.c_[XX.ravel(),YY.ravel()]
Z_pred = svdd.decision_function(Z)

# draw decision boundary 경계 표시
plt.figure(figsize=(10,8))
plt.contour(XX,YY,Z_pred.reshape(XX.shape), levels=[0], linewidths=2, colors='k')
# mark outliers (bounded support vectors)
plt.scatter(ring.iloc[bsv,0], ring.iloc[bsv,1], marker='x', color='blue')
# implement clusters 각 클러스터는 같은 색깔로 표시
plt.scatter(ring.iloc[label==0,0], ring.iloc[label==0,1], marker='o', s = 20, color = 'purple')
plt.scatter(ring.iloc[label==1,0], ring.iloc[label==1,1],marker='o',s = 20,color = 'green')
plt.scatter(ring.iloc[label==2,0], ring.iloc[label==2,1],marker='o',s = 20,color = 'dodgerblue')
plt.scatter(ring.iloc[label==3,0], ring.iloc[label==3,1],marker='o',s = 20,color = 'yellow')
