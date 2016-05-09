#!/usr/bin/python3

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
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from scipy import misc
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import RandomizedPCA
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split

			

# =================================
# function that converts images to high dimensional matrix with raw features
# it returns:
# 1. Data: a matrix: 1 - emotion;  2 - subjects;  3 - raw features
# 2. Label: a list of strings (emotions)
# =================================
def readImages(FileDir, orderFile, NumIllum, NumSub, normalized = False):

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
			if normalized:
				temp = temp - np.mean(temp)
				temp = temp / np.std(temp)
			Data[sub, illum, :] = temp

	# np.save(os.path.join(FileDir, 'Data_all_normalized'), Data)

	return Data


# =================================
# functions that reads a flattened vector and reshapes it into a image specified by width and heigh
# then it's saved as imagename
# =================================
def constructImages(Vector, width, height, imagename):

	pass


# =================================
# Extract eigenface features
# Input:
# 	FileDir - Directory for the images
# 	orderFile - order of images (order of illumination angles)
# 	NumIllum - number of illuminations
# 	NumSub - number of subjects
# 	TrainIndsFile - the indices of illuminations that are used as training set (although in each run leave one out from the training set for baseline)
# 	TestIndsFile - the indices of illuminations that are used as testing set
# 	ncomponents - number of components to compute
# 	Outdir - output directory
# 	randomized - boolean, if True use randomizedPCA, if False use PCA
# 	normalized - boolean, if the data is normalized by mean and standard error
# 	remove_dominant_components_out - integer, default 0, remove the first n dorminant components (remove the projections on those directions)
# Output:
# 	1. save the eigenface representation of each fold in OutDir
# 		3D matrix: 1 - subject (person);  2 - illumination; 3 - eigenface representation 
#	2. save the percentage of variance contributed by different components
# 	all saved as .npy
# =================================
def eigenfaceExtract(FileDir, orderFile, NumIllum, NumSub, TrainIndsFile, TestIndsFile, ncomponents, OutDir, randomized = True, normalized = False, remove_dominant_components_out = 0):

	# np arrays storing all the dimenion reduced features (projection coefficient on the dominant eigenfaces)
	# the axes are:
	# 1 - subject (person);  2 - illumination; 3 - eigenface representation 

	# # generate TestIndsFile
	# temp = []
	# for x in range(0, 64):
	# 	if x not in list(TrainInds.loc[:,0]):
	# 		temp.append(x)
	# temp = pd.DataFrame(temp)
	# temp.to_csv(TestIndsFile, header = None, index = None)

	TrainInds = pd.read_csv(TrainIndsFile, header = None)	# The illuminations that will be used as the file
	TestInds = pd.read_csv(TestIndsFile, header = None)

	# for saving percentage of variance ocntributed by different components
	EigenValuePercents = np.zeros((ncomponents, TrainInds.shape[0]))

	# choose randomizedPCA or PCA
	if randomized:
		print('Working on:', ncomponents, 'components with', 'randomizedPCA,', 'data normalized?', normalized, 'remove first several components?', remove_dominant_components_out)
	else:
		print('Working on:', ncomponents, 'components with', 'PCA,', 'data normalized?', normalized, 'remove first several components?', remove_dominant_components_out)

	# in each run, leave one out from training set to be used for baseline
	for fold in range(0,TrainInds.shape[0]):

		# In each run, leave one out from the training indices to be used for baseline
		TrainIndsTemp = []
		TrainIndsTemp.extend(list(TrainInds.loc[range(0,fold), 0]))		# checked! the values in TrainInds are integers, so they can be used as indices
		TrainIndsTemp.extend(list(TrainInds.loc[range(fold+1, TrainInds.shape[0]), 0]))
		TestIndsTemp = [TrainInds.loc[fold, 0]]				# The one that is left out is added to the first one in testing set				
		TestIndsTemp.extend(list(TestInds.loc[:,0]))
		TrainSize = len(TrainIndsTemp)						# TrainSize is now the original size - 1
		print('Train Inds:', TrainIndsTemp)
		print('Test Inds:', TestIndsTemp[0])

		# empty matrix for storing the low-dimensional features
		Features_train = np.zeros((NumSub, TrainSize, ncomponents - remove_dominant_components_out))				# first several components removed
		Features_test = np.zeros((NumSub, NumIllum-TrainSize, ncomponents - remove_dominant_components_out))

		# Read the images
		Data = readImages(FileDir, orderFile, NumIllum, NumSub, normalized)

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

		# perform dimension reduction on Train and Test
		Train_pca = pca.transform(Train)
		Test_pca = pca.transform(Test)

		# only select the rest of the components
		if remove_dominant_components_out > 0:
			Train_pca = Train_pca[:, remove_dominant_components_out:]
			Test_pca = Test_pca[:, remove_dominant_components_out:]

		# the percentaage of variance
		EigenValuePercents[:, fold] = pca.explained_variance_ratio_

		# plot and save the eigenfaces of the analysis
		# This part of code hasn't been optimized for compatibility
		print('plotting eigenfaces: ')
		h = 192
		w = 168			# this is the saize of images
		eigenfaces = pca.components_.reshape((ncomponents, h, w))
		eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
		# plot_gallery(eigenfaces, eigenface_titles, h, w)
		n_row = 2
		n_col = 3
		plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
		plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
		for i in range(n_row * n_col):
			plt.subplot(n_row, n_col, i + 1)
			plt.imshow(eigenfaces[i].reshape((h, w)), cmap=plt.cm.gray)
			plt.title(eigenface_titles[i], size=12)
			plt.xticks(())
			plt.yticks(())
		# plt.savefig('./test.png')
		plt.savefig(os.path.join(OutDir, 'eigenfaces_f' + str(fold) + '.png'))

		# get the time of PCA analysis for one frame
		print('PCA time for this fold: %.2fs' % (time.time() - t0))

		# after dimension reduction from PCA, the emotion and subject axes are seperated into a higher dimensional matrix
		for sub in range(0, NumSub):
			Features_train[sub,:,:] = Train_pca[sub*TrainSize:(sub+1)*TrainSize, :]
			Features_test[sub,:,:] = Test_pca[sub*(NumIllum-TrainSize):(sub+1)*(NumIllum-TrainSize), :]

		# save as the deault .npy binary file
		np.save(os.path.join(OutDir, 'egf_' + str(fold) + '_tr'), Features_train)
		np.save(os.path.join(OutDir, 'egf_' + str(fold) + '_te'), Features_test)

	# save the percentage of variance contirbuted by different components
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

	NumSub = 10
	NumIllum = 64
	ncomponents = [6,10]#,18,50,100]	# correspond to 80% 90% 95% >99% >>99%

	remove_dominant_components_out=0
	for ncomp in ncomponents:
		# randomizedPCA, unnormalized
		OutDir = os.path.join(WorkingDir, 'data/eigenfaces/unnormalized/' + str(ncomp) + '_component')
		if not os.path.exists(OutDir):
			os.makedirs(OutDir)
		eigenfaceExtract(FileDir, orderFile, NumIllum, NumSub, TrainIndsFile, TestIndsFile, ncomp, OutDir, randomized=True, normalized=False, remove_dominant_components_out=remove_dominant_components_out)
		# # PCA, unnormalized
		# OutDir = os.path.join(WorkingDir, 'data/eigenfaces_PCA/unnormalized/' + str(ncomp) + '_component')
		# if not os.path.exists(OutDir):
		# 	os.makedirs(OutDir)
		# eigenfaceExtract(FileDir, orderFile, NumIllum, NumSub, TrainIndsFile, TestIndsFile, ncomp, OutDir, randomized=False, normalized=False)
	


	ncomponents = [8,14]#,24,50,100]	# correspond to 80% 90% 95% >99% >>99%

	remove_dominant_components_out=0
	for ncomp in ncomponents:
		# randomizedPCA, normalized, remove_dominant_components_out=3
		OutDir = os.path.join(WorkingDir, 'data/eigenfaces/normalized/' + str(ncomp) + '_component')
		if not os.path.exists(OutDir):
			os.makedirs(OutDir)
		eigenfaceExtract(FileDir, orderFile, NumIllum, NumSub, TrainIndsFile, TestIndsFile, ncomp, OutDir, randomized=True, normalized=True, remove_dominant_components_out=remove_dominant_components_out)
		# # PCA, normalized, remove_dominant_components_out=3
		# OutDir = os.path.join(WorkingDir, 'data/eigenfaces_PCA/normalized/' + str(ncomp) + '_component')
		# if not os.path.exists(OutDir):
		# 	os.makedirs(OutDir)
		# eigenfaceExtract(FileDir, orderFile, NumIllum, NumSub, TrainIndsFile, TestIndsFile, ncomp, OutDir, randomized=False, normalized=True, remove_dominant_components_out=remove_dominant_components_out)

	remove_dominant_components_out=3
	for ncomp in ncomponents:
		# randomizedPCA, normalized, remove_dominant_components_out=3
		OutDir = os.path.join(WorkingDir, 'data/eigenfaces/normalized_first_' + str(remove_dominant_components_out) + '_components_removed/' + str(ncomp) + '_component')
		if not os.path.exists(OutDir):
			os.makedirs(OutDir)
		eigenfaceExtract(FileDir, orderFile, NumIllum, NumSub, TrainIndsFile, TestIndsFile, ncomp, OutDir, randomized=True, normalized=True, remove_dominant_components_out=remove_dominant_components_out)
		# # PCA, normalized, remove_dominant_components_out=3
		# OutDir = os.path.join(WorkingDir, 'data/eigenfaces_PCA/normalized_first_' + str(remove_dominant_components_out) + '_components_removed/' + str(ncomp) + '_component')
		# if not os.path.exists(OutDir):
		# 	os.makedirs(OutDir)
		# eigenfaceExtract(FileDir, orderFile, NumIllum, NumSub, TrainIndsFile, TestIndsFile, ncomp, OutDir, randomized=False, normalized=True, remove_dominant_components_out=remove_dominant_components_out)

if __name__ == '__main__':
    getFeatures()



