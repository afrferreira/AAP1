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
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.kde import KernelDensity

warnings.simplefilter(action='ignore',category=FutureWarning)

val = np.loadtxt('TP1-data.csv',delimiter = ',')

data = shuffle(val)

Ys = data[:,4]
Xs = data[:,:4]

means = np.mean(Xs,axis=0)

stdevs = np.std(Xs,axis=0)

Xs = (Xs-means)/stdevs

Xtrain,Xtest,Ytrain,Ytest = train_test_split(Xs, Ys, test_size=0.33, stratify=Ys)

"""Linear Regression"""
def calc_fold(feats, X,Y, train_ix, valid_ix,Cval):
    """return error for train and validation sets"""
    reg = LogisticRegression(C=Cval, tol=1e-10)
    reg.fit(X[train_ix, :feats],Y[train_ix])
    prob = reg.predict_proba(X[:,:feats])[:,1]
    squares = (prob - Y)**2
    return np.mean(squares[train_ix]),np.mean(squares[valid_ix])

folds = 5
kf = StratifiedKFold(n_splits = folds)
feats=4

    
def LogisticReg():
    minError=1
    Cval=1
    trainning_error = []
    validation_error = []
    for ite in range(20):
        tr_err = va_err = 0
        for tr_ix,va_ix in kf.split(Ytrain,Ytrain):
            r,v = calc_fold(feats,Xtrain,Ytrain,tr_ix,va_ix,Cval)
            tr_err += r
            va_err += v    
        trainning_error.append([np.log(Cval),tr_err/folds])
        validation_error.append([np.log(Cval),va_err/folds])
        Cval*=2
    trainning_error2 = np.array(trainning_error)
    validation_error2 = np.array(validation_error)
    plt.plot(trainning_error2[:,0],trainning_error2[:,1],'-r',label = 'training set')
    plt.plot(validation_error2[:,0],validation_error2[:,1],'-g', label='validation set') 
    plt.show()
    plt.close()
    if(va_err<minError):
        minError = va_err
    return minError

"""K-Nearest Neighbours"""

def calcNeighbor(X,Y, train_ix, valid_ix,k):
    knn = KNeighborsClassifier(k)
    knn.fit(X[train_ix, :feats],Y[train_ix])
    scoreTr = 1-knn.score(X[train_ix],Y[train_ix])
    scoreVal = 1-knn.score(X[valid_ix],Y[valid_ix])
    return scoreTr,scoreVal
    
def KNearest():
    minError = 1
    training_error = []
    validation_error = []
    for k in range(1,39,2):
        tr_err = va_err = 0
        for tr_ix,va_ix in kf.split(Ytrain,Ytrain):
            r,v = calcNeighbor(Xtrain,Ytrain,tr_ix,va_ix,k)
            tr_err +=r
            va_err += v
        training_error.append([k,tr_err/folds])
        validation_error.append([k,va_err/folds])
    trainer = np.array(training_error)
    valid = np.array(validation_error)
    plt.plot(trainer[:,0],trainer[:,1],'-r')
    plt.plot(valid[:,0],valid[:,1])
    plt.show()
    plt.close()
    if(va_err<minError):
        minError = va_err
    return minError

"""Naive Bayes"""

def scorer(log_z,z_kde,log_o,o_kde,test):
    classes = np.zeros(test.shape[0])
    for row in range(test.shape[0]):
        z_sum = log_z
        o_sum = log_o
        for column in range(test.shape[1]):
            z_sum = z_sum + z_kde[column].score_samples(test[row][column])
            o_sum = o_sum + o_kde[column].score_samples(test[row][column])
        if z_sum<o_sum:
            classes[row]=1
    return classes


def kdeCalc(one,b):
    kde=[]
    for x in range(one.shape[1]):
        kde.append(KernelDensity(bandwidth=b,kernel='gaussian').fit(one[:,x].reshape(one.shape[0],1)))
    return kde
        
def data_split(X,Y):
    zero_T_cat =np.array(X[Y==0,:])
    one_T_cat = np.array(X[Y==1,:])
    return zero_T_cat,one_T_cat

def baysian(X,Y,train_ix,valid_ix,b):
    z_train,o_train = data_split(X[train_ix],Y[train_ix])
    z_valid,o_valid = data_split(X[valid_ix],Y[valid_ix])
    z_kde = kdeCalc(z_train,b)
    o_kde = kdeCalc(o_train,b)
    tot_len = z_train.shape[0]+o_train.shape[0]
    log_z = np.log(z_train.shape[0]/tot_len)
    log_o = np.log(o_train.shape[0]/tot_len)
    classified_train = scorer(log_z,z_kde, log_o , o_kde , X[train_ix])
    classified_valid = scorer(log_z,z_kde, log_o , o_kde , X[valid_ix])
    train_err = 1-accuracy_score(Y[train_ix] , classified_train)
    valid_err = 1-accuracy_score(Y[valid_ix] , classified_valid)
    return train_err , valid_err

def naiveBayes():
    minError = 0
    training_error = []
    validation_error = []
    for b in range(1,100,2):
        tr_err = va_err = 0
        for tr_ix,va_ix in kf.split(Ytrain,Ytrain):
            r,v = baysian(Xtrain,Ytrain,tr_ix,va_ix,(b/100))
        
          
LogisticReg()
KNearest()
##naiveBayes()