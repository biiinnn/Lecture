# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 20:14:30 2022

@author: yebin
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch import nn, optim

import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'{device} is available.')

# GPU 사용 가능일 경우 랜덤 시드 고정
if device == 'cuda:0':
    torch.cuda.manual_seed_all(777)

#%% 학습에 사용할 파라미터 설정
# 기본 변수
input_size = 28*28 # image size = 28*28 (흑백이미지)
num_classes = 10

# 학습 파라미터
batch_size = 100
learning_rate = 0.001
training_epochs = 20

#%% 데이터 준비
# 이미지를 텐서로 변환 (0~255 -> 0~1)
transform  = transforms.Compose([transforms.ToTensor()])

# 데이터 불러오기
trainset = datasets.FashionMNIST(root='./.data/',
                                 train=True,
                                 download=True,
                                 transform=transform)
testset = datasets.FashionMNIST(root='./.data/',
                                train=False,
                                download=True,
                                transform=transform)

# 데이터 로드
trainloader = DataLoader(trainset,
                         batch_size=batch_size, 
                         shuffle=True,
                         drop_last=True) # 마지막 배치 버리기
testloader = DataLoader(testset,
                        batch_size=batch_size,
                        shuffle=False,
                        drop_last=True) # 마지막 배치 버리기

#%% Logistic Regression
class LogReg(nn.Module):
    
    def __init__(self):
        super(LogReg, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
   
    def forward(self, x):
        x = self.linear(x)
        return x

#%% FNN1 - BatchNormaliztion 사용
class FNN1(nn.Module):
    
    def __init__(self):
        super(FNN1, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512), # 데이터 중간중간 normalize
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 64),
            nn.BatchNorm1d(64), # 데이터 중간중간 normalize
            nn.ReLU()
        )
        self.fc = nn.Linear(in_features=64, out_features=num_classes)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.fc(x)
        return x
    
#%% FNN2 - Dropout 사용
class FNN2(nn.Module):
    
    def __init__(self):
        super(FNN2, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(0.25) # 오버피팅을 막기 위해 학습 과정 시 일부 생략
        self.fc = nn.Linear(in_features=64, out_features=num_classes)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

#%% FNN3 - BatchNormaliztion, Dropout 사용
class FNN3(nn.Module):
    
    def __init__(self):
        super(FNN3, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512), # 데이터 중간중간 normalize
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 64),
            nn.BatchNorm1d(64), # 데이터 중간중간 normalize
            nn.ReLU()
        )
        self.dropout = nn.Dropout(0.25) # 오버피팅을 막기 위해 학습 과정 시 일부 생략
        self.fc = nn.Linear(in_features=64, out_features=num_classes)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

#%% 모델 생성
model1 = LogReg().to(device)
model2 = FNN1().to(device)
model3 = FNN2().to(device)
model4 = FNN3().to(device)

# 모델 구조 출력
print(model1)
print(model2)
print(model3)
print(model4)

# loss function 정의
criterion = nn.CrossEntropyLoss().to(device) # CrossEntropyLoss는 softmax계산까지 포함되어 있음

#%% 모델 학습
def train_model(model):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) #옵티마이저 정의
    
    loss_ = [] # 그래프를 그리기 위한 loss 저장용 리스트
    n = len(trainloader) # 배치 개수
    
    print('Start Training')
    for epoch in range(training_epochs): # 20번 학습 진행
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            images = images.view(-1, input_size).requires_grad_().to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images) #예측값 산출
            loss = criterion(outputs, labels) # 손실함수 계산
            loss.backward() # 손실함수 기준으로 역전파 선언
            optimizer.step() # 가중치 최적화
            
            running_loss += loss.item()
        loss_.append(running_loss/n)
        
        print('[%d] loss: %.4f' %(epoch + 1, running_loss / n)) # 1에폭마다 평균 loss값 출력
    print('Finished Training')
    return loss_
#%% 학습 진행
loss1 = train_model(model1)
loss2 = train_model(model2)
loss3 = train_model(model3)
loss4 = train_model(model4)

#%% train loss 그래프 그리기
plt.plot(loss1, label='LogReg')
plt.plot(loss2, label='FNN1')
plt.plot(loss3, label='FNN2')
plt.plot(loss4, label='FNN3')
plt.title("Training Loss")
plt.legend()
plt.xlabel("epoch")
plt.show()

loss1
#%% 모델 평가 
def test_model(model):
    pred = [] #예측 값 저장용 리스트
    t_loss = [] # test_loss 저장용 리스트
    t_acc = [] # 정확도 저장용 리스트
    n = len(testloader) # 배치 개수
    
    test_loss = 0.0
    total = 0
    correct = 0
    
    with torch.no_grad():
        model.eval()
        for images, labels in testloader:
            images = images.view(-1, 28*28).requires_grad_().to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            test_loss = loss.item()
            t_loss.append(test_loss/n)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0) # 총 개수
            correct += (predicted == labels).sum().item() # 맞으면 1, 틀리면 0으로 합산
            accuracy = 100 * correct / total
            
            pred.append(predicted)
            t_acc.append(accuracy)

    mean_accuracy = np.mean(t_acc)

    return t_loss, mean_accuracy, pred

#%% test data 평가 진행
t_loss1, acc1, pred1 = test_model(model1)
t_loss2, acc2, pred2 = test_model(model2) 
t_loss3, acc3, pred3 = test_model(model3) 
t_loss4, acc4, pred4 = test_model(model4)   
#%% test loss 그래프 그리기
plt.plot(t_loss1, label='LogReg')
plt.plot(t_loss2, label='FNN1')
plt.plot(t_loss3, label='FNN2')
plt.plot(t_loss4, label='FNN3')
plt.title("Test Loss")
plt.legend()
plt.xlabel("batch")
plt.show()

#%% 정확도 출력
print('Logistic Regression Accuray: ', round(acc1,4))
print('FNN1 Accuray: ', round(acc2,4))
print('FNN2 Accuray: ', round(acc3,4))
print('FNN3 Accuray: ', round(acc4,4))

#%% confusion matrix 그리는 함수 
# https://towardsdatascience.com/exploring-confusion-matrix-evolution-on-tensorboard-e66b39f4ac12

def plot_confusion_matrix(cm, class_names):
    
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure
# confusion matrix 그리기
def cm(pred):
    m_predict = []
    for i in range(len(pred)):
        for p in pred[i].tolist():
            m_predict.append(p)
    
    conf_matrix=confusion_matrix(testset.targets.tolist(), m_predict)
    plot_confusion_matrix(conf_matrix, testset.classes)

#%% 가장 성능이 좋은 모델 CM 찍기
cm(pred4)

 
