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
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
print(sigmoid(0)) #0.5
x = np.linspace(-np.pi, np.pi, 10) # checking if it works for vectors
print(sigmoid(x))

# adding the intercept term 
m=len(y)
import statsmodels.api as sm
X=np.append(arr=np.ones((m,1)).astype(int),values=X,axis=1)
theta = np.zeros((X.shape[1], 1))
# x.theta
def multtheta(x,theta):
    return np.dot(x,theta)
# calculating the sigmoid of the product
def calcsigmoid(x,theta):
    return sigmoid(multtheta(x,theta))
#Cost function and gradient
def computeCost(x,y,theta):
    totalcost = -(1 / m) * np.sum(
        y * np.log(calcsigmoid(x,theta)) + (1 - y) * np.log(
            1 - calcsigmoid( x,theta)))
    return totalcost

def calcgradient(x,y,theta):
    