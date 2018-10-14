#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 09:41:16 2018

@author: andrefrf
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings;
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from t3_aux import poly_16features, create_plot

warnings.simplefilter(action='ignore',category=FutureWarning)

val = np.loadtxt('TP1-data.csv',delimiter = ',')

data = shuffle(val)

Ys = data[:,4]
Xs = data[:,:4]

means = np.mean(Xs,axis=0)

stdevs = np.std(Xs,axis=0)

Xs = (Xs-means)/stdevs

Xr,Xt,Yr,Yt = train_test_split(Xs, Ys, test_size=0.33, stratify=Ys)

def calc_fold(feats, X,Y, train_ix, valid_ix,Cval):
    """return error for train and validation sets"""
    reg = LogisticRegression(C=Cval, tol=1e-10)
    reg.fit(X[train_ix, :feats],Y[train_ix])
    prob = reg.predict_proba(X[:,:feats])[:,1]
    squares = (prob - Y)**2
    return np.mean(squares[train_ix]),np.mean(squares[valid_ix])

folds = 5
kf = StratifiedKFold(n_splits = folds)
for feats in range(1,5):
   tr_err = va_err = 0
   for tr_ix,va_ix in kf.split(Yr,Yr):
       Cval=1
       for ite in range(1,21):    
           Cval*=20
           r,v = calc_fold(feats,Xr,Yr,tr_ix,va_ix,Cval)
           tr_err += r
           va_err += v
           tr = tr_err/folds
           va = va_err/folds
           print(feats,':', tr,va)
           plt.figure(1,figsize=(30,10),frameon=False)
           plt.plot(tr, va ,'or',color = "C"+str(feats))
           
           
plt.savefig("tp1LogReg.png")  
plt.close()         