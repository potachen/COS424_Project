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

def prepareDataInput( imageArray ):
    '''Take in a 1D array of images and return a matrix
    with dimensions nImages x nPixels'''
    imageArraySize = np.shape(imageArray)[0]
    nPixels = np.size(imageArray[0])
    matrix = np.zeros((imageArraySize,nPixels))
    # Fill each row of the matrix with a 1D array of one image
    for index in range(imageArraySize):
        matrix[index,:] = imageArray[index].flatten(order='C')
    return matrix

def produceEncoding( trainX, nComponents ):
    '''Produces an NMF encoding from the training
    data matrix'''
    model = NMF( n_components=nComponents, solver='cd', \
                tol=1e-4, max_iter=200, alpha=0.0 )
    model.fit( trainX )
    return model

def prepareDataOutput( lowDimMatrix ):
    '''Take in a matrix nSubjects x nComponents
    and output an array: nSubject x nIlluminations
    x nComponents'''
    return lowDimMatrix

def reduceDim( trainData, testData, nComponents ):
    '''Takes in an image for each subject and returns
    a low dimensional representation for each subject '''
    trainX = prepareDataInput( trainData )
    testX = prepareDataInput( testData )
    model = produceEncoding( trainX, nComponents )
    lowDimTrainData = model.transform( trainX )
    lowDimTestData = model.transform( testX )
    lowDimTrainOutput = prepareDataOutput( lowDimTrainData )
    lowDimTestOutput = prepareDataOutput( lowDimTestData )
    return lowDimTrainData, lowDimTestData

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
