# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 23:35:09 2021

@author: jpg99
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score


class perceptron():
    def __init__(self,max_iter= 1000,learning_rate = 0.3):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        
    def initialize(self,x):
        self.n = x.shape[0]
        self.m = x.shape[1]
        self.w = np.ones(self.m)  #np.random.rand(1,self.m)
        self.b = 0      #np.random.rand()
    def model(self, x):
        return 1 if (np.dot(self.w, x) >= self.b) else 0
  
  
    def predict(self, X):
        pred = []
        for x in X:
            result = self.model(x)
            pred.append(result)
        return np.array(pred) 
        
        
    def train(self,x_train,y_train):
        for i,j in zip(x_train,y_train):
            pred = self.model(i)
            self.w = self.w + self.learning_rate*(j-pred)*i
            self.b = self.b - self.learning_rate*(j-pred)*1
            
    def fit(self,X,Y):
        self.initialize(X)
        for i in range(self.max_iter):
            self.train(X,Y)
        pass

    

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


data1=pd.read_csv("ls_data\Class1.txt",names=['a','b'])
data1['class']=1

data2=pd.read_csv("ls_data\Class2.txt",names=['a','b'])
data2['class']=0
plt.scatter(data1['a'],data1['b'])
plt.scatter(data2['a'],data2['b'])
plt.show

data=pd.concat([data1,data2])
y=data['class']
data=data.drop(['class'], axis=1)
X=data.to_numpy()
y=y.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
scaler = MinMaxScaler()
scaler.fit(X_train)
X=scaler.transform(X)
X_train=scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = perceptron()
model.fit(X_train,y_train)

pred = model.predict(X_test)

print("accuracy: "+str(accuracy_score(y_test,pred)*100)+"%")
print(confusion_matrix(y_test,pred))

fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface ')
# Set-up grid for plotting.
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.99)
#ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
for i in range(len(pred)):
    if pred[i]==1:
        ax.scatter(X_test[i][0],X_test[i][1],color='r', cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    if pred[i]==0:
        plt.scatter(X_test[i][0],X_test[i][1],color='b',cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)

plt.show()

