#!/usr/local/bin/python

'''
COS424 project: Recognising facial expressions
Po-Ta Chen, Sagar Setru, Hugh Wilson, Zidong Zhang
Module: Feature extraction using Non Negative Matrix Factorisation
    INPUT
        An nxm training data matrix: n: pixel index; m: image index.
        An nxt testing data matrix
    RETURN
        An rxm encoded training data matrix: r: low dimensional component index;
        An rxt encoded testing data matrix
'''

import pandas as pd
import numpy as np
import math
import sys
import os
import time
from scipy import ndimage
from scipy import misc
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import NMF
import sklearn.datasets as data

def prepareDataInput( inArray ):
    '''Take in a 3D n x m x p array and return
    a 2D (nxm) x p array'''
    nSize = np.shape( inArray )[0]
    mSize = np.shape( inArray )[1]
    pSize = np.shape( inArray )[2]
    outArray = np.zeros( (nSize*mSize, pSize) )
    for i in range(nSize):
        for j in range(mSize):
            outArray[(j+i*mSize),:] = inArray[i,j,:]
    return outArray

def produceEncoding( trainX, nComponents, alpha, l1_ratio ):
    '''Produces an NMF encoding from the training
    data matrix'''
    model = NMF( n_components=nComponents, solver='cd', \
                tol=1e-4, max_iter=200, alpha=alpha, \
                l1_ratio=l1_ratio )
    model.fit( trainX )
    return model

def prepareDataOutput( inArray, out0, out1 ):
    '''Take in a 2D (nxm) x p array and return
    a 3D n x m x p array. n: out0, m: out1'''
    nSize = out0
    mSize = out1
    pSize = np.shape( inArray )[1]
    outArray = np.zeros( (nSize,mSize,pSize) )
    for i in range(nSize):
        for j in range(mSize):
            outArray[i,j,:] = inArray[(j+i*mSize),:]
    return outArray

def reduceDim( trainData, testData, nComponents, alpha, l1_ratio ):
    '''Takes in an image for each subject and returns
    a low dimensional representation for each subject '''
    outDim0 = np.shape( trainData )[0]
    trainOutDim1 = np.shape( trainData )[1]
    testOutDim1 = np.shape( testData )[1]
    trainX = prepareDataInput( trainData )
    testX = prepareDataInput( testData )
    model = produceEncoding( trainX, nComponents, alpha, l1_ratio )
    lowDimTrainData = model.transform( trainX )
    lowDimTestData = model.transform( testX )
    lowDimTrainOutput = prepareDataOutput( lowDimTrainData, \
                                          outDim0, trainOutDim1 )
    lowDimTestOutput = prepareDataOutput( lowDimTestData, \
                                         outDim0, testOutDim1 )
    return lowDimTrainOutput, lowDimTestOutput, \
        model.reconstruction_err_, model.components_

def main():
    nComponents = 50
    # Import a dataset for testing
    Faces = data.fetch_olivetti_faces()
    Images = Faces.images
    trainData = Images[:100,:,:]
    testData = Images[100:,:,:]
    # Produce a low dimensional representation
    lowDimTrainData, lowDimTestData = reduceDim( trainData, testData, \
                                                nComponents )


if __name__ == '__main__':
    main()
