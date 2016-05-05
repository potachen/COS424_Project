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
from scipy import ndimage
from scipy import misc
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import RandomizedPCA
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split

# =================================
# class for the image data
# 1. data: a matrix with each row as an image
# 2. label: a vector with emotion labels
# =================================
class ImageData(object):
	def __init__(self, data, label):
		if data.shape[0] != len(label):
			raise ValueError('data and emotion labels label are not in the same size!')
		else:
			self.data = data
			self.label = label


# =================================
# class for the dimension reduced features
# =================================
class DimReducedFeature(object):
	def __init__(self, train, test, label):
		if train.shape[0] != test.shape[0]:
			raise ValueError('training and testing data are different in number of frames')
		elif train.shape[1] != test.shape[1]:
			raise ValueError('training and testing data are different in number of emotions')
		elif train.shape[3] != test.shape[3]:
			raise ValueError('training and testing data are different in number of features')
		else:
			self.train = train
			self.test = test
			self.label = label
			

# =================================
# function that converts images to high dimensional matrix with raw features
# it returns:
# 1. Data: a matrix: 1 - emotion;  2 - subjects;  3 - raw features
# 2. Label: a list of strings (emotions)
# =================================
def readImages(FileDir, FrameInd, emotionLab, NumSub, Imagesize):

	# A np array storing raw image features for a single time point
	# the axes are:
	# 1 - emotion;  2 - subjects;  3 - raw features
	# Data = np.random.rand(len(emotionLab), NumSub, Imagesize)			# random numbers for testing
	Data = np.zeros((len(emotionLab), NumSub, Imagesize))
	
	# Imagesize ??
	# Read images
	for emo in range(0, len(emotionLab)):
		for k in range(0, NumSub):
			temp = misc.imread(FileDir)
			if len(face.shape) != 2:
				raise ValueError('Image is not black and white')
			
			temp = temp.flatten()
			Data[emo, k, :] = temp
	
	return ImageData(Data, emotionLab)


# =================================
# Extract eigenface features
# Input: 
# Output:
# =================================
def eigenfaceExtract(FileDir, NumFrame, emotionLab, NumSub, Imagesize, ncomponents):

	# np arrays storing all the dimenion reduced features (projection coefficient on the dominant eigenfaces)
	# the axes are:
	# 1 - time;  2 - expression (emotion);  3 - subject (person);  4 - eigenface/NMF representation 
	NumEmo = len(emotionLab)
	cv_fold = 0.25
	NumTest = math.ceil(cv_fold * NumSub)	# number of subjects for testing
	NumTrain = NumSub - NumTest				# number of subjects for training
	Features_train = np.zeros((NumFrame, NumEmo, NumTrain, ncomponents))
	Features_test = np.zeros((NumFrame, NumEmo, NumTest, ncomponents))

	for t in range(0, NumFrame):
		images = readImages(FileDir, t, emotionLab, NumSub, Imagesize)

		# np arrays for the raw features of all emotions and subjects at one single point
		# emotions and subjects are merged into one axis to perform PCA on them
		Train = np.zeros((NumEmo*NumTrain, Imagesize))
		Test = np.zeros((NumEmo*NumTest, Imagesize))
		
		for emo in range(0, len(emotionLab)):
			temp = images.data[emo, :, :]

			# transform the images data into 
			temp_train, temp_test = train_test_split(temp, test_size=0.25)

			Train[emo*NumTrain:(emo+1)*NumTrain, :] = temp_train
			Test[emo*NumTest:(emo+1)*NumTest, :] = temp_test


		# get the time of PCA analysis for one frame
		t0 = time.time()

		# get the eigenfaces with from all the data points (all emotions and subjects at one single time point) in the training set
		# ????????????
		# use PCA or randomizedPCA ?
		# ????????????
		# pca = PCA(n_components=ncomponents, whiten=True).fit(Train)
		pca = RandomizedPCA(n_components=ncomponents, whiten=True).fit(Train)

		# perform dimension reduction on Train and X_test
		Train_pca = pca.transform(Train)
		Test_pca = pca.transform(Test)

		# get the time of PCA analysis for one frame
		print('PCA time for this frame: %.2fs' % (time.time() - t0))

		# after dimension reduction from PCA, the emotion and subject axes are seperated into a higher dimensional matrix
		for emo in range(0, len(emotionLab)):
			Features_train[t,emo,:,:] = Train_pca[emo*NumTrain:(emo+1)*NumTrain, :]
			Features_test[t,emo,:,:] = Test_pca[emo*NumTest:(emo+1)*NumTest, :]

	return DimReducedFeature(Features_train, Features_test, emotionLab)


# =================================
# Main function
# =================================
def getFeatures():
	WorkingDir = '/media/zidong/Work/Princeton/COS424/project'
	cv_num = 1		# number of cross vadiations to be performed

	ImageDir = os.path.join(WorkingDir, 'data/test_eigenface')
	if not os.path.exists(ImageDir):
		os.makedirs(ImageDir)

	FeatureDir = os.path.join(WorkingDir, 'data/test_eigenface')
	if not os.path.exists(FeatureDir):
		os.makedirs(FeatureDir)

	NumFrame = 2
	emotionLab = ['happy', 'sad']
	NumSub = 50
	Imagesize = 10000
	ncomponents = 50

	for i in range(0, cv_num):
		Features = eigenfaceExtract(ImageDir, NumFrame, emotionLab, NumSub, Imagesize, ncomponents)
		np.save(FeatureDir + '/' + '-'.join(Features.label) + 'train_' + str(i), Features.train)
		np.save(FeatureDir + '/' + '-'.join(Features.label) + 'test_' + str(i), Features.train)



