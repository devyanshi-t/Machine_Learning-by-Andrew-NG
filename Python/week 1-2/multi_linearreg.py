
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('ex1data2.txt')
X=dataset.iloc[:,0:2]
y=dataset.iloc[:,2:3]
m=len(y)
# Feature Normalization
#Subtract the mean value of each feature from the dataset.
#After subtracting the mean, additionally scale (divide) the feature values by their respective “standard deviations.”
avg=np.mean(X)
sd=np.std(X)
X=(X-avg)/sd

import statsmodels.api as sm
X=np.append(arr=np.ones((m,1)).astype(int),values=X,axis=1)
alpha=0.01
i=1000

# cost function
theta=np.zeros([3,1])

def computemulticost(X,y,theta):
    t=np.dot(X,theta)-y
    return np.sum(np.power(t,2))/(2*m)
    
J=computemulticost(X,y,theta)
print(J.item) # 6.527919e+10
 # vector to store min cost
# Gradient Descent
def mgradientdescent(X,y,theta,alpha,step):
    for i in range(step):
        t1=np.dot(X,theta)-y
        t2=np.dot(X.T,t1)
    return theta-(alpha*t2)/m
theta=mgradientdescent(X,y,theta,alpha,i)
print(theta)
J = computemulticost(X, y, theta)
print(J.item)
# Selecting learning rates\
alpha=0.009
for j in range(4):
  #alpha=np.array([0.1,0.3,0.03,0.01])
  i=np.array([25,100,275,450,890])
  theta=mgradientdescent(X,y,theta,alpha,i[j])
  temp = computemulticost(X, y, theta)
  print(J.item)
  J=np.append(J,temp)
  
plt.plot(i,J,color='red') #plotting graph for cost vs number of iteration
plt.title(' no.of iteration vs cost')
plt.xlabel('no.of iteration')
plt.ylabel('Cost')
plt.show()


# 6.399107e+10

predictval = ([1,16.5,3]) * theta;




# Normal EquationsX
from numpy.linalg import inv
a=np.array(X)
part=np.dot(a.transpose(),a)
part1 = np.dot(inv(part),a.transpose())
final=np.dot(part1,y)

predictvalne = ([1,16.5,3]) * final;

