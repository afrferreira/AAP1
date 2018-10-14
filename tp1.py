#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 09:41:16 2018

@author: andrefrf
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from t3_aux import poly_16features, create_plot


val = np.loadtxt('TP1-data.csv',delimiter = ',')

data = shuffle(val)

Ys = data[:,4]
Xs = data[:,:4]

means = np.mean(Xs,axis=0)

stdevs = np.std(Xs,axis=0)

Xs = (Xs-means)/stdevs

Xr,Xt,Yr,Yt = train_test_split(Xs, Ys, test_size=0.33, stratify=Ys)