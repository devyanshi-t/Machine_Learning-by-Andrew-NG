#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 17:58:16 2019

@author: devyanshitiwari
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv('ex2data1.txt')
X=dataset.iloc[:,[0,1]]
y=dataset.iloc[:,2]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.25,random_state=0)

# Visualizing the data
admitted = dataset.loc[y == 1]

not_admitted = dataset.loc[y == 0]

plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted',marker='+',color='black')
plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted',color='red')
plt.legend()
plt.xlabel('exam 1 score')
plt.ylabel('exam 2 score')
plt.show()

# Warmup exercise: sigmoid function
def sigmoid(X):
    return 1 / (1 + np.exp(-X))
    
print(sigmoid(0)) #0.5
temp = np.linspace(-np.pi, np.pi, 10) # checking if it works for vectors
print(sigmoid(temp))

# adding the intercept term 
m=len(y)
import statsmodels.api as sm
X=np.append(arr=np.ones((m,1)).astype(int),values=X,axis=1)
y = y[:, np.newaxis]
theta = np.zeros([3, 1])
# x.theta
def multtheta(X,theta):
    return np.dot(X,theta)
# calculating the sigmoid of the product
def calcsigmoid(X,theta):
    return sigmoid(multtheta(X,theta))
#Cost function and gradient
def computeCost(theta,X,y):
    totalcost = -(1 / m) * np.sum(
        y * np.log(calcsigmoid(X,theta)) + (1 - y) * np.log(
            1 - calcsigmoid( X,theta)))
    return totalcost
# setting alpha
alpha=0.01
step=400
def calcgradient(theta,X,y):
    for i in range(step):
        t1=calcsigmoid(X,theta)
        t1=t1-y
        t2=np.dot(X.T,t1)
    return theta-(alpha*t2)/m
theta=calcgradient(theta,X,y)
print(theta)
J = computeCost( theta,X, y)
print(J)  # 0.6931471805599454


import scipy.optimize as op
t = op.fmin_tnc(func=computeCost ,x0=theta.flatten(),fprime=calcgradient,
                 args=(X, y.flatten()))
t_opti=t[0]
J = computeCost( t_opti[:,np.newaxis],X, y)
print(J)

                
                   
               
# plotting the decision boundary
predict1 = [45, 85] * theta;
