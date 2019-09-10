
"""
Created on Tue Sep 10 12:00:38 2019

@author: devyanshitiwari
"""
# Step 1 convert the text file into csv file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('data1.csv')
X=dataset.iloc[:,0:1]
y=dataset.iloc[:,1:2]

m=len(y) # number of training examples
# visualising the data by creating a scatter plot
plt.scatter(X,y,color='red')
plt.title('Population VS Profit')
plt.xlabel('Population in 10,000')
plt.ylabel('Profit in $10,000s')
plt.show()

 
# adding the intercept term and setting alpha=0.01
import statsmodels.api as sm
X=np.append(arr=np.ones((m,1)).astype(int),values=X,axis=1) # Adding the intercept

# calculating cost function for theta =0

alpha=0.01
theta=np.zeros([2,1])

def computecost(X,y,theta):
    t=np.dot(X,theta) -y
    return np.sum(np.power(t,2))/(2*m)
    
J=computecost(X,y,theta)
print(J.item) # 30.975

# Gradient Descent
# checking for convergence of gradient descent

def gradientdescent(X,y,theta,alpha,step):
    for i in range(step):
        t1=np.dot(X,theta)-y
        t2=np.dot(X.T,t1)
    return theta-(alpha*t2)/m

    
for i in range(1000):
 theta=gradientdescent(X,y,theta,alpha,1000)
 print(theta)
 J = computecost(X, y, theta)
 print(J.item) # it was observed that value of J was decreasing until still started to converge

# predicting the values
predict1 = [1, 3.5] * theta;
predict2 = [1, 7] * theta;
print(predict1,predict2)

# plotting the line of best fit
plt.scatter(X[:,1],y,color='red')
plt.plot(X[:,1],np.dot(X,theta),color='blue')
plt.title('Population VS Profit')
plt.xlabel('Population in 10,000')
plt.ylabel('Profit in $10,000s')
plt.show()


