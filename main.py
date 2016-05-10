#!/usr/local/bin/python


'''
COS424 project: Recognising facial expressions
Po-Ta Chen, Sagar Setru, Hugh Wilson, Zidong Zhang
Hugh's version of the main file.
Used for loading in the data and executing the nmf
dimension reduction on each fold
'''

import pandas as pd
import numpy as np
import FEMs.NMF.lnmf as NMF
import matplotlib.pyplot as plt

def reshapeAndPrint( components, fold, alphaVal, l1_ratio ):
    '''Takes in the vector encoding of each of the nmf components
    and plots them to file'''
    h = 192
    w = 168
    nComp = np.shape( components )[0]
    nCompPlot = nComp #6
    plotComps = components
    plotComps.reshape((nCompPlot,h,w))
    titles = ["comp. %d" % (i+1) for i in range(plotComps.shape[0])]
    n_row = 4
    n_col = 6
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0.03, left=.01, right=.99, top=.97, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(plotComps[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    # save
    plt.savefig('./Results/alpha' + str(alphaVal) + '/l1ratio' + str(l1_ratio) + \
                '/Images/nComp' + str(nComp) + \
             '_fold' + str(fold) + '.png')


def main():
    # Dataset parameters
    nIllCond = 64
    # Saving parameters
    alphaVals = [10]     #[25,50]    #[1,10,100]
    l1Vals = [0.5]

    # Read in the data structure
    rawImages = np.load('./Data/Data_all_normalized.npy')
    # Translate so that no values are negative
    rawImages -= np.min( rawImages )

    # Read in indices of the Test and Train (TAT) and Train only (T0) Datasets
    tatIndDf = pd.read_csv('./Data/TrainingIllums.txt', header=None)
    toIndDf = pd.read_csv('./Data/TestingIllums.txt', header=None)
    allIndDf = pd.DataFrame(range(nIllCond))
    # Make these into lists
    tatInd = [i for i in tatIndDf[0]]
    toInd = [i for i in toIndDf[0]]
    allInd = [i for i in allIndDf[0]]

    # Split into Test and Train (TAT) and Train only (TO)
    tatImages = rawImages[:,tatInd,:]
    toImages = rawImages[:,toInd,:]
    allImages = rawImages[:,allInd,:]

    # Produce a low dimensional representation for each
    # fold of the TAT datasubset
    sizeTat = np.shape(tatInd)[0]
    nSubjects = np.shape(tatImages)[0]
    nPixels = np.shape(allImages)[2]
    compList = [2,8,14,24,50]
    # Record the reconstruction error for each nComponents and fold
    sizeCompList = np.shape(compList)[0]
    recErrorMat = np.zeros((sizeCompList,sizeTat))
    for l1_ratio in l1Vals:
        for alphaVal in alphaVals:
            compIndex = -1
            for nComp in compList:
                compIndex += 1
                for fold in range(sizeTat):
                    # Produce the training dataset
                    tempTatImages = np.zeros((nSubjects,sizeTat-1,nPixels))
                    tatCol = 0
                    for tempCol in range(sizeTat-1):
                        if fold == tempCol:
                            tatCol += 1
                        tempTatImages[:,tempCol,:] = tatImages[:,tatCol,:]
                        tatCol += 1
                    # Produce the testing dataset
                    sizeTo = nIllCond - sizeTat + 1
                    tempToImages = np.zeros((nSubjects,sizeTo,nPixels))
                    toCol = 0
                    for tempCol in range(sizeTo):
                        if toCol == 0:
                            tempToImages[:,0,:] = tatImages[:,fold,:]
                        else:
                            tempToImages[:,toCol,:] = toImages[:,toCol-1,:]
                        toCol += 1
                    lowDimTrain, lowDimTest, recErr, compnts = NMF.reduceDim( tempTatImages, \
                                                            tempToImages, \
                                                            nComponents=nComp, \
                                                            alpha=alphaVal, \
                                                            l1_ratio=l1_ratio )
                    np.save('./Results/alpha' + str(alphaVal) + \
                            '/l1ratio' + str(l1_ratio) + '/' + str(nComp) + \
                            '/nmf_' + str(fold) + '_tr', lowDimTrain )
                    np.save('./Results/alpha' + str(alphaVal) + \
                            '/l1ratio' + str(l1_ratio) + '/' + str(nComp) + \
                            '/nmf_' + str(fold) + '_te', lowDimTest )
                    recErrorMat[compIndex,fold] = recErr
                    #if fold == 0:
                        #reshapeAndPrint( compnts, fold, alphaVal, l1_ratio )

            # Write the reconstruction error to file
            np.save('./Results/alpha' + str(alphaVal) + \
                    '/l1ratio' + str(l1_ratio) + '/recError',recErrorMat)



if __name__ == '__main__':
    main()
