#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 17:31:44 2018

@author: pili
"""

import os
import math
import pandas as pd

def readEmail(filename):
    """input: name of a the txt file
        output: list of words of the message in the email
    """
    #data = [filename]
    data = []
    try:
        fh = open(filename,'r',encoding = 'latin1')
    except IOError:
        print('cannot open', filename)
    else:
        for line in fh:
            if line !='\n':
                words =  line[:-1].split(' ')
                for word in words:
                    data.append(word)
    finally:
        fh.close()
    return data

def readDirectory(directory, class_name):
    '''Takes the name of the directory and the name of the class
        and creates a subset of data  
    '''
    d = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            d.append([readEmail(os.path.join(root,name)),class_name])
    return d

def extractVocabulary(samples):
    """
    """
    vocabulary = []
    for sample in samples:
        #print (sample)
        for word in sample[0]:
            #print(word)
            if word not in vocabulary:
                vocabulary.append(word)
    return vocabulary

def split_70_30(samples):
    total_samples = len(samples)
    split_limit = int(total_samples*7/10)
    f = []
    l = []
    counter = 0
    for sample in samples:
        if counter <= split_limit:
            f.append(sample)
            counter += 1
        else:
            l.append(sample)
            counter += 1
    return f,l 

def filter_data_by_class(samples,class_name):
    result = []
    for sample in samples:
        if sample[1] == class_name:
            result.append(sample)
    return result

def split_training_data(classes_names,samples):
    first70 = []
    last30 = []
    for class_name in classes_names:
        filtered_data_by_class = filter_data_by_class(samples,class_name)
        #print(len(filtered_data_by_class))
        f,l = split_70_30(filtered_data_by_class)
        first70 = first70 + f
        last30 = last30 + l
    return first70,last30


def createDataFrame(vocabulary,samples):
    len_samples = len(samples)
    #print(len_samples)
    result= pd.DataFrame(0,index = range(len_samples),columns = vocabulary)
    for sample_index in range(len_samples):
        #print(sample_index)
        for word in samples[sample_index][0]:
            if word in vocabulary:
                result[word][sample_index]=1
    return result

def createDataStructure(vocabulary,samples):
    X_matrix = []
    Y_vector = []
    for sample in samples:
        Xl = [1]
        Yl = sample[1]
        for word in vocabulary:
            if word in sample[0]:
                Xl.append(1)
            else:
                Xl.append(0)
        X_matrix.append(Xl)
        Y_vector.append(Yl)
    return X_matrix, Y_vector

    
def createXvector(vocabulary,message):
    X=[1]
    for word in vocabulary:
        if word in message:
            X.append(1)
        else:
            X.append(0)
    return X

def dotProductXlW(Xl,W):
    if len(Xl) == len(W):
        result = 0
        for i in range(len(W)):
            result += Xl[i]*W[i]
    else:
        print('wrong dotProductXlW')
    return result

def zig_log_fun(x):
    """Numerically-stable sigmoid function."""
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        # if x is less than zero then z will be small, denom can't be
        # zero because it's 1+z.
        z = math.exp(x)
        return z / (1 + z)

def prob_yl_is_1_givenX(Xl,W):
    WX = dotProductXlW(Xl,W)
    result = zig_log_fun(WX)
    return result

def compute_new_W(W,X,Y,eta,lambda_MCAP):
    n = len(W)
    new_W = []
    for i in range(n):
        temp_w = W[i]
        for l in range(len(X)):
            #print("(",l,",",i,")", end =" " )
            temp_w += eta * X[l][i] * (Y[l] - prob_yl_is_1_givenX(X[l],W))
        temp_w -= eta * lambda_MCAP * W[i]
        new_W.append(temp_w)
    return new_W 


def Converge_W(W_init,X,Y,eta,lambda_MCAP,iterarions):
    W_old = W_init.copy()
    for i in range(1,iterarions+1):
        print("Interation i: ")
        temp = compute_new_W(W_old,X,Y,eta,lambda_MCAP)
        W_old = temp.copy()
    return W_old



""" ------------------ Testing functionality ------------------ """

train_ham_file = '/home/pili/T2/dataset_1/train/ham'
train_spam_file = '/home/pili/T2/dataset_1/train/spam'
test_ham_file = '/home/pili/T2/dataset_1/test/ham'
test_spam_file = '/home/pili/T2/dataset_1/test/spam'

data_train = readDirectory(train_spam_file,1) + readDirectory(train_ham_file,0) 
C = [0,1]

samples_70_train,samples_30_validation = split_training_data(C,data_train)
V = extractVocabulary(samples_70_train)
len_of_W = len(V)+1
W_init = [0]*len_of_W
X, Y = createDataStructure(V,samples_70_train)
eta = 0.1
lambda_MCAP = 0
iterarions = 10

result = Converge_W(W_init,X,Y,eta,lambda_MCAP,iterarions)


