#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 16:46:58 2019

@author: devyanshitiwari
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('ex1data2.txt')
X=dataset.iloc[:,0:2]
y=dataset.iloc[:,2]

# Feature Normalization

# Gradient Descent
# Selecting learning rates
# Normal Equations
