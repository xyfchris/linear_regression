# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 17:16:28 2017

@author: I310036
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

# Dataset is included in sklearn
from sklearn.datasets import load_boston

# Load dataset
boston = load_boston()

# boston is a dictionary, showing the keys
boston.keys()

# hyper parameters
alpha = 0.01 # learning rate
num_epoch = 1000000

x = boston.data
y = boston.target
x.shape

x = x / x.max(axis=0)  #Normalization

m = len(y)   # number of examples
y = y.reshape(m,1)

# Build design matrix, which is x plus a '1' column in the first position
x1 = np.insert(x,0,1,axis=1)  # insert a column of '1's into the first position of x
x1[:1, :]  # first row

theta = np.matrix(np.zeros((14,1)))  # initialize theta

 def grad_descent(x1,y,theta,iter_num,alpha):
    m = len(y)
    costs=[]
    for i in range(iter_num):

        theta += x1.T*(y - x1*theta)*alpha/m
        cost = 1/2/m * (x1*theta-y).T*(x1*theta-y)
        if (i%10000)==0: 
            costs.append(cost)
            print('iteration = %i, cost = %.8f' %(i,cost)) 
    return theta, costs