# -*- coding: utf-8 -*-

# DO NOT CHANGE
import pandas as pd
import numpy as np

def get_order(structure):
    # structure: dictionary of structure
    #            key is variable and value is parents of a variable
    # return list of learning order of variables 
    # ex) ['A', 'R', 'E', 'S', 'T', 'O']
    result = []
    
    # 부모 노드가 없는 경우 (루트인 경우) => 추가
    for k, v in structure.items():
        if len(v) == 0:
            result.append(k)
    # 부모 노드가 있는 경우1 - 부모 노드가 하나일 때, 순서 리스트와 부모 노드가 일치 => 추가
    for k, v in structure.items():
        if v == result:
            result.append(k)
    # 부모 노드가 있는 경우2 - 부모 노드가 하나일 때, 순서 리스트에 부모가 없음 
    #                        순서 리스트에 이미 루트가 되는 노드들을 추가하였기 때문에 경우2는 발생X
    # 부모 노드가 있는 경우3 - 부모 노드가 여러 개일 때, 순서 리스트에 모든 부모가 있고 자신은 없음 => 자신 추가
    for k, v in structure.items():
        if set(v).issubset(set(result)):
            if k not in result:
                result.append(k)
    # 부모 노드가 있는 경우4 - 부모 노드가 여러 개일 때, 순서 리스트에 없는 부모가 있음 => 부모 노드 추가
    for k, v in structure.items():
        for p in v:
            if p not in result:
                result.append(p)
    # 부모 노드가 있는 경우5 - 경우4에서 부모를 모두 추가 하고 마지막으로 자신 추가
    for k, v in structure.items():
        if set(v).issubset(set(result)):
            if k not in result:
                result.append(k)
    
    return result

def learn_parms(data,structure,var_order):
    # data: training data
    # structure: dictionary of structure
    # var_order: list of learning order of variables
    # return dictionary of trained parameters (key=variable, value=learned parameters)
    result = dict()
    
    for var in var_order:
        # 부모 노드가 없는 경우
        if len(structure[var]) == 0:
            df1 =  pd.DataFrame(data[var].value_counts(normalize=True).sort_index()).T
            df1.columns = np.unique(data[var])
            df1.reset_index(drop=True, inplace=True)
            result[var] =  df1
            continue
        # 부모 노드가 있는 경우
        p_index = []
        for p in structure[var]:
            p_index.append(data[p]) # 부모 노드의 데이터 인덱스 조합
        # 부모 노드의 state 조합에 따른 확률 분포
        df2 = pd.DataFrame(pd.crosstab(p_index, data[var], normalize='index'))
        df2.columns = np.unique(data[var])
        result[var] = df2
        
    return result

def print_parms(var_order,parms):
    # var_order: list of learning order of variables
    # parms: dictionary of trained parameters (key=variable, value=learned parameters)
    # print the trained parameters for each variable
    for var in var_order:
        print('-------------------------')
        print('Variable Name=%s'%(var))
        #TODO: print the trained paramters
        if len(parms[var]) == 1:
            print(parms1[var].to_string(index=False))
        else:
            print(parms[var])
    
data=pd.read_csv('https://drive.google.com/uc?export=download&id=1taoE9WlUUN4IbzDzHv7mxk_xSj07f-Zt', sep=' ')

str1={'A':[],'S':[],'E':['A','S'],'O':['E'],'R':['E'],'T':['O','R']}
order1=get_order(str1)
parms1=learn_parms(data,str1,get_order(str1))
print('-----First Structure------')
print_parms(order1,parms1)
print('')

str2={'A':['E'],'S':['A','E'],'E':['O','R'],'O':['R','T'],'R':['T'],'T':[]}
order2=get_order(str2)
parms2=learn_parms(data,str2,get_order(str2))
print('-----Second Structure-----')
print_parms(order2,parms2)
print('')


