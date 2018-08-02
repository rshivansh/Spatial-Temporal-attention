import numpy as np
import os
import os.path as osp
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch

from PIL import Image
from skimage.color import rgb2yuv
import cv2
import pdb


#return all images 
def prepareDataset(datasetRootDir, datasetRootDirOF, fileExt, dataset_name, disableOpticalFlow):
        personDirs = getPersonDirsList(datasetRootDir,dataset_name)
	nPersons = len(personDirs)
	dataset = [0 for _ in xrange(nPersons)]
	letter = ['a','b']
	for i,pdir in enumerate(personDirs):
		dataset[i] = [0 for _ in xrange(len(letter))]
		for cam in xrange(len(letter)):
			cameraDirName=''
			if dataset_name == 1:
                		cameraDirName = 'cam'+str(cam+1)            
            		else:
                		cameraDirName = 'cam_'+str(letter[cam])
			seqRoot = osp.join(datasetRootDir,cameraDirName,pdir)
			seqRootOF = osp.join(datasetRootDirOF,cameraDirName,pdir)
			seqImgs = getSequenceImageFiles(seqRoot,fileExt) 
			dataset[i][cam] = loadSequenceImages(seqRoot,seqRootOF,seqImgs,disableOpticalFlow)

		#only use first 200 persons who appear in both cameras for PRID 2011
        	if dataset_name == 2 and i == 200:
            		return dataset
	return dataset



#get a sorted list of directories for all the persons in the dataset
def getPersonDirsList(seqRootDir, dataset_name):
	firstCameraDirName=""
	if dataset_name == 1:
        	firstCameraDirName = 'cam1'
        else:
        	firstCameraDirName = 'cam_a'
	tmpSeqCam = osp.join(seqRootDir,firstCameraDirName)
	personDirs = []

	#Go over all files in directory
	for filename in os.listdir(tmpSeqCam):
		personDirs.append(filename)

	#Check files exist
	assert len(personDirs) is not None

        return sorted(personDirs)

#given a directory containing all images in a sequence get all the image filenames in order
def getSequenceImageFiles(seqRoot,filesExt):
	seqFiles = []
	#Go over all files in directory
	for filename in os.listdir(seqRoot):
		#We only load files that match the extension
		if filename.endswith(filesExt):
			seqFiles.append(filename)
	assert len(seqFiles) is not None
	seqFiles=sorted(seqFiles)
	return seqFiles

#load all images into a flat list
def loadSequenceImages(cameraDir,opticalflowDir, filesList, disableOpticalFlow):
	scaler = transforms.Scale((48, 64))
	#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
	to_tensor = transforms.ToTensor()
	nImgs = len(filesList)
	for i,file_name in enumerate(filesList):
		filename = osp.join(cameraDir,file_name)
		filenameOF = osp.join(opticalflowDir,file_name)
		img = cv2.imread(filename)
		img = cv2.resize(img,(48,64)) 
		imgof = cv2.imread(filenameOF)
		imgof = cv2.resize(imgof,(48,64))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
		img_tensor=to_tensor(img)
		imgof_tensor=to_tensor(imgof) 
		if i==0:
			imagePixelData=torch.FloatTensor(nImgs,5,img_tensor.shape[1],img_tensor.shape[2])
		for c in xrange(to_tensor(img).shape[0]):	
			v = torch.sqrt(torch.Tensor([torch.var(img_tensor[c])])) 
			m = torch.mean(img_tensor[c]) 
			img_tensor[c] = img_tensor[c] - m
			img_tensor[c] = img_tensor[c] / torch.sqrt(v)
			imagePixelData[i, c, :,:] = img_tensor[c] 
		for c in xrange(2):
			index=c+1
			v = torch.sqrt(torch.Tensor([torch.var(imgof_tensor[index])]))
			m = torch.mean(imgof_tensor[index])
			imgof_tensor[index] = imgof_tensor[index] - m
			imgof_tensor[index] = imgof_tensor[index] / torch.sqrt(v)
			imagePixelData[i, c+3, :,:] = imgof_tensor[index]
			if disableOpticalFlow is 'true':
				imagePixelData[i, c+3, :,:].mul(0)
		#imagePixelData[i]=img_tensor
	return imagePixelData
