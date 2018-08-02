import torch
from torch.autograd import Variable
from datasetUtils import getPosSample,getNegSample,prepare_person_seq
import timeit
import torch.nn as nn
from test import computeCMC_MeanPool_RNN
import torchvision.transforms as transforms
from PIL import Image
import math
import cv2
import pdb

def train_test_Sequence(model, dataset, trainInds, testInds, nEpochs,sampleSeqLength,lr,momentum,samplingEpochs):
	nTrainPersons = len(trainInds)
	model.cuda()
	model.train()
	optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
	for eph in range(1, nEpochs+1): 
		batchErr  = 0
		order = torch.randperm(nTrainPersons)
		tic=timeit.default_timer()
		for i in xrange((nTrainPersons*2)):
			if i % 2 == 0:
			#choose a positive pair, both sequences show the same person
				pushPull = 1
		        	camA = 0
		        	camB = 1
				startA,startB,seq_length = getPosSample(dataset,trainInds,order[i/2], sampleSeqLength)
				netInputA = dataset[trainInds[order[i/2]]][camA][startA:(startA + seq_length),:,:,:].squeeze()
				netInputB = dataset[trainInds[order[i/2]]][camB][startB:(startB + seq_length),:,:,:].squeeze()
				netTarget = [1,(order[i/2]),(order[i/2])]
			else:
			#choose a negative pair, both sequences show different persons
				pushPull = -1
				seqA,seqB,camA,camB,startA,startB,seq_length = getNegSample(dataset,trainInds,sampleSeqLength)
				netInputA = dataset[trainInds[seqA]][camA][startA:(startA + seq_length),:,:,:].squeeze()
				netInputB = dataset[trainInds[seqB]][camB][startB:(startB + seq_length),:,:,:].squeeze()
				netTarget = [-1,seqA,seqB]

			#set the parameters for data augmentation. Note that we apply the same augmentation
            		#to the whole sequence, rather than individual images
			crpxA = torch.floor(torch.rand(1).squeeze() * 8) + 1
			crpyA = torch.floor(torch.rand(1).squeeze() * 8) + 1
			crpxB = torch.floor(torch.rand(1).squeeze() * 8) + 1
        		crpyB = torch.floor(torch.rand(1).squeeze() * 8) + 1
        		flipA = torch.floor(torch.rand(1).squeeze() * 2) + 1
        		flipB = torch.floor(torch.rand(1).squeeze() * 2) + 1
			
			netInputA = doDataAug(netInputA,crpxA,crpyA,flipA)
            		netInputB = doDataAug(netInputB,crpxB,crpyB,flipB)

			netInputA=Variable(netInputA).cuda()
			netInputB=Variable(netInputB).cuda()
			
			optimizer.zero_grad()
			#loss function
			hinge_criterion = nn.HingeEmbeddingLoss()
			
			nl_criterion1 = nn.NLLLoss()
			nl_criterion2 = nn.NLLLoss()
			#pass image data to model
			feature_vec1, logsoftmax1=model(netInputA)
			feature_vec2, logsoftmax2=model(netInputB)
			
			#calculate l2 pairwise distance
			distance=nn.PairwiseDistance(p=2)
			target1=Variable(torch.Tensor([netTarget[0]]).cuda())

			target2 = Variable(torch.LongTensor([netTarget[1]]).cuda()) 
			target3 = Variable(torch.LongTensor([netTarget[2]]).cuda())
			#import ipdb;ipdb.set_trace()
			dist=distance(feature_vec1,feature_vec2)
			loss=hinge_criterion(dist,target1) + nl_criterion1(logsoftmax1,target2)+ nl_criterion2(logsoftmax2,target3)
			
			#print('Loss : %.4f'%loss.data[0])
			batchErr=batchErr+loss

			loss.backward()
			optimizer.step()

		toc=timeit.default_timer()
		if eph % 1 == 0:
            		print('Epoch: %d, Batch Error: %.4f, Time=%.4f '%(eph, batchErr.data[0], toc - tic))
            		batchErr = 0
		
		if eph==1300:
			lr=lr/10
			optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

		#import ipdb; ipdb.set_trace()
		if (eph % samplingEpochs == 0):
			cmcTest,simMatTest = computeCMC_MeanPool_RNN(dataset,testInds,model,sampleSeqLength)
			cmcTrain,simMatTrain = computeCMC_MeanPool_RNN(dataset,trainInds,model,sampleSeqLength)

			outStringTest =  'Test  '
            		outStringTrain = 'Train '
			printInds = [0,1,2,3,4,5,6,7,8,9,10]

			for c in xrange(len(printInds)):
				if c < nTrainPersons:		
					outStringTest = outStringTest + str(int(math.floor(cmcTest[printInds[c]]))) +' '
                    			outStringTrain = outStringTrain + str(int(math.floor(cmcTrain[printInds[c]])))+ ' '

			print(outStringTest)
            		print(outStringTrain)	

			model.train()

	return model
			

def doDataAug(seq,cropx,cropy,flip):
	seqLen = seq.shape[0]
        seqChnls = seq.shape[1]
        seqDim1 = seq.shape[2]
        seqDim2 = seq.shape[3]
	cropx=int(cropx[0])
	cropy=int(cropy[0])
	flip=flip[0]
	
	daData = torch.zeros(seqLen,seqChnls,seqDim1-8,seqDim2-8)
	#to_pillow_image=transforms.ToPILImage()
	to_tensor = transforms.ToTensor()
	for i in xrange(seqLen): 
		#import ipdb;ipdb.set_trace()
		#do the data augmentation here
		thisFrame = seq[i,:,:,:].squeeze().clone()
		if flip == 1:
			"""pil_image=to_pillow_image(thisFrame)
			flip_image=pil_image.transpose(Image.FLIP_LEFT_RIGHT)
            		thisFrame=to_tensor(flip_image)"""
			img_np = thisFrame.numpy()
			img_np = cv2.flip(img_np,0)
			thisFrame=torch.from_numpy(img_np) #to_tensor(img_np)
			
		#pdb.set_trace()
		thisFrame = thisFrame[:, cropx: (56 + cropx), cropy:(40 + cropy)]
		thisFrame = thisFrame - torch.mean(thisFrame)
		daData[i,:,:,:] = thisFrame
	
	return daData	
			
	
	
