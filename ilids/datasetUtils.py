import numpy as np
import os
import torch
from torch.autograd import Variable
import cv2
import torchvision.transforms as transforms
from PIL import Image
import pdb


def partitionDataset(nTotalPersons,testTrainSplit):
	splitPoint = int(nTotalPersons * testTrainSplit)
	inds = torch.randperm(nTotalPersons)
	trainInds = inds[0:splitPoint]
	testInds = inds[splitPoint:nTotalPersons]

	print('N train = '+str(trainInds.shape[0]))
        print('N test  = '+str(testInds.shape[0]))

	return trainInds,testInds

def feature_extraction_using_vgg(feature_net,dataset):
	nPersons = len(dataset)
	extracted_feature = [0 for _ in xrange(nPersons)]

	#set the parameters for data augmentation. Note that we apply the same augmentation
	#to the whole sequence, rather than individual images
	crpxA = torch.floor(torch.rand(1).squeeze() * 8) + 1
	crpyA = torch.floor(torch.rand(1).squeeze() * 8) + 1
	crpxB = torch.floor(torch.rand(1).squeeze() * 8) + 1
        crpyB = torch.floor(torch.rand(1).squeeze() * 8) + 1
        flipA = torch.floor(torch.rand(1).squeeze() * 2) + 1
        flipB = torch.floor(torch.rand(1).squeeze() * 2) + 1

	letter = ['a','b']
	for i in xrange(nPersons):
		extracted_feature[i] = [0 for _ in xrange(len(letter))]
		for cam in xrange(len(letter)):
			#pdb.set_trace()
			if cam==0:
				data=doDataAug(dataset[i][cam],crpxA,crpyA,flipA)
			else:
				data=doDataAug(dataset[i][cam],crpxB,crpyB,flipB)
			input_video=Variable(data,volatile=True)
			video_feature=feature_net(input_video.cuda())
			extracted_feature[i][cam]=video_feature
	return extracted_feature

def getPosSample(dataset,trainInds,person,sampleSeqLen):
	#choose the camera, ilids video only has two, but change this for other datasets
	camA = 0
    	camB = 1
    	actualSampleSeqLen = sampleSeqLen
	nSeqA = len(dataset[trainInds[person]][camA])
	nSeqB = len(dataset[trainInds[person]][camB])

	#what to do if the sequence is shorter than the sampleSeqLen 
	if nSeqA <= sampleSeqLen or nSeqB <= sampleSeqLen:
		if nSeqA < nSeqB:
			actualSampleSeqLen = nSeqA
		else:
            		actualSampleSeqLen = nSeqB

	startA = int(torch.rand(1)[0] * ((nSeqA - actualSampleSeqLen) + 1))
	startB = int(torch.rand(1)[0] * ((nSeqB - actualSampleSeqLen) + 1)) 

	return startA,startB,actualSampleSeqLen

def getNegSample(dataset,trainInds,sampleSeqLen):
	permAllPersons = torch.randperm(len(trainInds))
	personA = permAllPersons[0]
	personB = permAllPersons[1]
	#choose the camera, ilids video only has two, but change this for other datasets
	camA = int(torch.rand(1)[0] * 2) #+1
	camB = int(torch.rand(1)[0] * 2) #+1

	actualSampleSeqLen = sampleSeqLen
	nSeqA = len(dataset[trainInds[personA]][camA]) #len(dataset[trainInds[personA]][camA-1])
    	nSeqB = len(dataset[trainInds[personB]][camB]) #len(dataset[trainInds[personB]][camB-1])

	#what to do if the sequence is shorter than the sampleSeqLen 
	if nSeqA <= sampleSeqLen or nSeqB <= sampleSeqLen:
		if nSeqA < nSeqB:
            		actualSampleSeqLen = nSeqA
        	else:
            		actualSampleSeqLen = nSeqB

	startA = int(torch.rand(1)[0] * ((nSeqA - actualSampleSeqLen) + 1)) #+ 1
	startB = int(torch.rand(1)[0] * ((nSeqB - actualSampleSeqLen) + 1)) #+ 1

	return personA,personB,camA,camB,startA,startB,actualSampleSeqLen

def prepare_person_seq(personImages,trainInds,order,cam,start,seq_length):
	#pdb.set_trace()
	person_seq_original=torch.FloatTensor(seq_length,128)
	for i in xrange(seq_length): 
		person_seq_original[i]=personImages[trainInds[order]][cam][start+i].data.cpu()
	person_seq=person_seq_original.transpose(0,1)
	person_seq=person_seq.unsqueeze(0)
	person_seq=person_seq.unsqueeze(2)
	return person_seq_original, person_seq

#perform data augmentation to a sequence of images stored in a torch tensor
"""def doDataAug(seq,cropx1,cropy1,flip1):
	flip=int(flip1[0])
	cropx=int(cropx1[0])
	cropy=int(cropy1[0])
	seqLen = seq.shape[0]
        seqChnls = seq.shape[1]
        seqDim1 = seq.shape[2]
        seqDim2 = seq.shape[3]
	
	daData = torch.zeros(seqLen,seqChnls,seqDim1-8,seqDim2-8)
	to_pillow_image=transforms.ToPILImage()
	to_tensor = transforms.ToTensor()
	for i in xrange(seqLen): 
		#do the data augmentation here
		thisFrame = seq[i,:,:,:].squeeze().clone()
		#pdb.set_trace()
		if flip == 1:
			pil_image=to_pillow_image(thisFrame)
			flip_image=pil_image.transpose(Image.FLIP_LEFT_RIGHT)
            		thisFrame=to_tensor(flip_image)
		thisFrame = thisFrame[:, cropx: (56 + cropx), cropy:(40 + cropy)]
		thisFrame = thisFrame - torch.mean(thisFrame)
		daData[i,:,:,:] = thisFrame
	
	return daData"""

