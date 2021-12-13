# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 23:43:50 2021

@author: jpg99
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier

data1=pd.read_csv("nls_data\Class1.txt",names=['a','b'])
data1['class']=1

data2=pd.read_csv("nls_data\Class2.txt",names=['a','b'])
data2['class']=2
plt.scatter(data1['a'],data1['b'])
plt.scatter(data2['a'],data2['b'])
plt.show

data=pd.concat([data1,data2])
y=data['class']
data=data.drop(['class'], axis=1)
X=data.to_numpy()
y=y.to_numpy()
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
scaler = MinMaxScaler()
scaler.fit(x_train)
X=scaler.transform(X)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

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

#linear kernel
svm1 = SVC(kernel = 'linear')
svm1.fit(x_train,y_train)
pred = svm1.predict(x_test)
print("accuracy: "+str(accuracy_score(y_test,pred)*100)+"%")
print(confusion_matrix(y_test,pred))

fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface')
# Set-up grid for plotting.
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, svm1, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
#ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
for i in range(len(pred)):
    if pred[i]==1:
        ax.scatter(x_test[i][0],x_test[i][1],color='b', cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    if pred[i]==2:
        plt.scatter(x_test[i][0],x_test[i][1],color='r',cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)

plt.show()

#polynomial kernel
svm2 = SVC(kernel = 'poly')
svm2.fit(x_train,y_train)
pred2 = svm2.predict(x_test)
print("accuracy: " + str(accuracy_score(y_test,pred2)*100) + "%")
print(confusion_matrix(y_test,pred2))

fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface')
# Set-up grid for plotting.
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, svm2, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
#ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
for i in range(len(pred2)):
    if pred2[i]==1:
        ax.scatter(x_test[i][0],x_test[i][1],color='b', cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    if pred2[i]==2:
        plt.scatter(x_test[i][0],x_test[i][1],color='r',cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)

plt.show()

#rbf kernel
svm3 = SVC(kernel = 'rbf')
svm3.fit(x_train,y_train)

pred3 = svm3.predict(x_test)
print("accuracy: " + str(accuracy_score(y_test,pred3)*100)+ "%")
print(confusion_matrix(y_test,pred3))

fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface')
# Set-up grid for plotting.
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, svm3, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
#ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
for i in range(len(pred3)):
    if pred3[i]==1:
        ax.scatter(x_test[i][0],x_test[i][1],color='b', cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    if pred3[i]==2:
        plt.scatter(x_test[i][0],x_test[i][1],color='r',cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)

plt.show()


mlp1 = MLPClassifier(random_state = 2,max_iter = 1000)
mlp1.fit(x_train,y_train)
pred4 = mlp1.predict(x_test)
print("accuracy: " + str(accuracy_score(y_test,pred4)*100)+ "%")
print(confusion_matrix(y_test,pred4))

fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface')
# Set-up grid for plotting.
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, mlp1, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
#ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
for i in range(len(pred4)):
    if pred4[i]==1:
        ax.scatter(x_test[i][0],x_test[i][1],color='b', cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    if pred4[i]==2:
        plt.scatter(x_test[i][0],x_test[i][1],color='r',cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)

plt.show()

