########################################
# This is a module for reading images and extracting features based on eigenfaces
# eigenfaces are determined from the training images
# the projection coeficients are calculated for both training and testing images
# 
# 
# Zidong Zhang
# 2016-May-1
########################################

import pandas as pd
import numpy as np
import math
import sys
import os
import time
from PIL import Image
from scipy import ndimage
from scipy import misc
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import RandomizedPCA
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split


# # =================================
# # class for the dimension reduced features
# # =================================
# class DimReducedFeature(object):
# 	def __init__(self, train, test, label):
# 		if train.shape[0] != test.shape[0]:
# 			raise ValueError('training and testing data are different in number of frames')
# 		elif train.shape[1] != test.shape[1]:
# 			raise ValueError('training and testing data are different in number of emotions')
# 		elif train.shape[3] != test.shape[3]:
# 			raise ValueError('training and testing data are different in number of features')
# 		else:
# 			self.train = train
# 			self.test = test
# 			self.label = label
			

# =================================
# function that converts images to high dimensional matrix with raw features
# it returns:
# 1. Data: a matrix: 1 - emotion;  2 - subjects;  3 - raw features
# 2. Label: a list of strings (emotions)
# =================================
def readImages(FileDir, orderFile, NumIllum, NumSub):

	# A np array storing raw image features for a single time point
	# the axes are:
	# 1 - subjects;  2 - illuminations;  3 - raw features
	Order = pd.read_csv(orderFile, header = None)
	Imagesize = len(misc.imread(os.path.join(FileDir, Order.loc[0, 0])).flatten())
	Data = np.zeros((NumSub, NumIllum, Imagesize))

	# # check: every subject has the same illuminations and they are in the same order
	# for k in range(0, 64):
	# 	temp = []
	# 	for j in range(0,10):
	# 		temp.append(Order.loc[j*64+k ,0][7:])
	# 	temp = set(temp)
	# 	if len(temp) > 1:
	# 		print('Wrong order in illumination: ' + str(k))

	# Read images
	for sub in range(0, NumSub):
		for illum in range(0, NumIllum):
			File = os.path.join(FileDir, Order.loc[NumIllum*sub + illum, 0])
			temp = misc.imread(File).flatten()
			Data[sub, illum, :] = temp

	# np.save(os.path.join(FileDir, 'Data_all_raw'), Data)

	return Data


# =================================
# Extract eigenface features
# Input: 
# Output:
# =================================
def eigenfaceExtract(FileDir, orderFile, NumIllum, NumSub, TrainIndsFile, TestIndsFile, ncomponents, OutDir, randomized):

	# np arrays storing all the dimenion reduced features (projection coefficient on the dominant eigenfaces)
	# the axes are:
	# 1 - subject (person);  2 - illumination; 3 - eigenface/NMF representation 

	# temp = []
	# for x in range(0, 64):
	# 	if x not in list(TrainInds.loc[:,0]):
	# 		temp.append(x)
	# temp = pd.DataFrame(temp)
	# temp.to_csv('./data/TestingIllums.txt', header = None, index = None)

	TrainInds = pd.read_csv(TrainIndsFile, header = None)	# The illuminations that will be used as the file
	TestInds = pd.read_csv(TestIndsFile, header = None)

	EigenValuePercents = np.zeros((ncomponents, TrainInds.shape[0]))

	if randomized:
		print('Working on:', ncomponents, 'components with', 'randomizedPCA')
	else:
		print('Working on:', ncomponents, 'components with', 'PCA')

	# in each run, 
	for fold in range(0,TrainInds.shape[0]):

		# In each run, leave one out from the training indices to be used for baseline
		TrainIndsTemp = []
		TrainIndsTemp.extend(list(TrainInds.loc[range(0,fold),0]))		# checked! the values in TrainInds are integers
		TrainIndsTemp.extend(list(TrainInds.loc[range(fold+1, TrainInds.shape[0]),0]))
		TrainSize = len(TrainIndsTemp)
		TestIndsTemp = [fold]
		TestIndsTemp.extend(list(TestInds.loc[:,0]))

		# empty matrix for storing the low-dimensional features
		Features_train = np.zeros((NumSub, TrainSize, ncomponents))				# leave one out from the training data for calculating the baseline
		Features_test = np.zeros((NumSub, NumIllum-TrainSize, ncomponents))

		# Read the images
		Data = readImages(FileDir, orderFile, NumIllum, NumSub)

		# randomizedPCA in scikit learn needs the input to be a 2D long matrix
		# need to concatenate subjects and illuminations into one dimension
		Train = np.zeros((NumSub*TrainSize, Data.shape[2]))
		Test = np.zeros((NumSub*(NumIllum-TrainSize), Data.shape[2]))
			
		for sub in range(0, NumSub):
			Train[sub*TrainSize:(sub+1)*TrainSize, :] = Data[sub, TrainIndsTemp, :]
			Test[sub*(NumIllum-TrainSize):(sub+1)*(NumIllum-TrainSize), :] = Data[sub, TestIndsTemp, :]


		# get the time of PCA analysis for one frame
		t0 = time.time()

		# get the eigenfaces with from all the data points (all emotions and subjects at one single time point) in the training set
		# ????????????
		# use PCA or randomizedPCA ?
		# ????????????
		if randomized:
			pca = RandomizedPCA(n_components=ncomponents, whiten=True).fit(Train)
		else:
			pca = PCA(n_components=ncomponents, whiten=True).fit(Train)

		# perform dimension reduction on Train and X_test
		Train_pca = pca.transform(Train)
		Test_pca = pca.transform(Test)

		EigenValuePercents[:, fold] = pca.explained_variance_ratio_

		# get the time of PCA analysis for one frame
		print('PCA time for this fold: %.2fs' % (time.time() - t0))

		# after dimension reduction from PCA, the emotion and subject axes are seperated into a higher dimensional matrix
		for sub in range(0, NumSub):
			Features_train[sub,:,:] = Train_pca[sub*TrainSize:(sub+1)*TrainSize, :]
			Features_test[sub,:,:] = Test_pca[sub*(NumIllum-TrainSize):(sub+1)*(NumIllum-TrainSize), :]

		np.save(os.path.join(OutDir, 'egf_' + str(fold) + '_tr'), Features_train)
		np.save(os.path.join(OutDir, 'egf_' + str(fold) + '_te'), Features_test)

	np.savetxt(os.path.join(OutDir, 'Percents.txt'), EigenValuePercents, fmt='%.5f', delimiter='\t')



# =================================
# Main function
# =================================
def getFeatures():
	WorkingDir = '/media/zidong/Work/Princeton/COS424/project'
	FileDir = os.path.join(WorkingDir, 'data/background_subtracted/tif_files')
	orderFile = os.path.join(WorkingDir, 'data/MasterFileOrder.txt')

	TrainIndsFile = os.path.join(WorkingDir, 'data/TrainingIllums.txt')
	TestIndsFile = os.path.join(WorkingDir, 'data/TestingIllums.txt')

	# ImageDir = os.path.join(WorkingDir, 'data/test_eigenface')
	# if not os.path.exists(ImageDir):
	# 	os.makedirs(ImageDir)

	NumSub = 10
	NumIllum = 64
	ncomponents = [6,10,20,50,100]

	for ncomp in ncomponents:
		OutDir = os.path.join(WorkingDir, 'data/eigenfaces/' + str(ncomp) + '_component')
		if not os.path.exists(OutDir):
			os.makedirs(OutDir)
		eigenfaceExtract(FileDir, orderFile, NumIllum, NumSub, TrainIndsFile, TestIndsFile, ncomp, OutDir, randomized=True)

		OutDir = os.path.join(WorkingDir, 'data/eigenfaces_PCA/' + str(ncomp) + '_component')
		if not os.path.exists(OutDir):
			os.makedirs(OutDir)
		eigenfaceExtract(FileDir, orderFile, NumIllum, NumSub, TrainIndsFile, TestIndsFile, ncomp, OutDir, randomized=False)


if __name__ == '__main__':
    getFeatures()



