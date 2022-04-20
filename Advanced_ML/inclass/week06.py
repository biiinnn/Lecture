# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 10:57:27 2021

@author: yebin
"""

# HIll Climb Search
import pandas as pd
import numpy as np

data=pd.DataFrame(np.random.randint(0,5,size=(5000,9)), columns=list('ABCDEFGHI'))
data['J']=data['A']*data['B']

from pgmpy.estimators.HillClimbSearch import HillClimbSearch
from pgmpy.estimators import BicScore

est=HillClimbSearch(data, scoring_method=BicScore(data))
best_model=est.estimate()

sorted(best_model.nodes())
best_model.edges() # edge가 어디에 세워졌는지 확인 가능

est.estimate(max_indegree=1).edges()

# Structure Learning
from pgmpy.models.BayesianModel import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling

student=BayesianModel([('diff','grade'),('intel','grade')])
cpd_d=TabularCPD('diff',2,[[0.6],[0.4]])
cpd_i=TabularCPD('intel',2,[[0.7],[0.3]])
cpd_g=TabularCPD('grade',3,[[0.3,0.05,0.9,0.5],[0.4,0.25,0.08,0.3],[0.3,0.7,0.02,0.2]],['intel','diff'],[2,2])

print(cpd_g)

student.add_cpds(cpd_d, cpd_i, cpd_g)

inference=BayesianModelSampling(student)
samples=inference.forward_sample(200)

est=HillClimbSearch(samples, scoring_method=BicScore(samples))
best_model=est.estimate()

best_model.edges()

from pgmpy.estimators.BayesianEstimator import BayesianEstimator #베이지안 파라미터 추정

best_model2=BayesianModel(best_model.edges())
best_model2.edges()
best_model2.nodes()
pest=BayesianEstimator(best_model2, samples)

print(pest.estimate_cpd('grade'))