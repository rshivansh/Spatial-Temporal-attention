import torch
from torch.autograd import Variable
import timeit
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import math
import cv2
import pdb

#standard method of computing the CMC curve using sequences
def computeCMC_MeanPool_RNN(dataset,cmcTestInds,net,sampleSeqLength):
	personImgs=dataset
	net.eval()
	nPersons = len(cmcTestInds)

	avgSame = 0
	avgDiff = 0
	avgSameCount = 0
	avgDiffCount = 0

	simMat = torch.zeros(nPersons,nPersons)
	to_pillow_image=transforms.ToPILImage()
	to_tensor = transforms.ToTensor()
	for shiftx in xrange(7):
		for doflip in xrange(1):
			shifty = shiftx
			feats_cam_a = torch.DoubleTensor(nPersons,128) 
			for i in xrange(nPersons): 
				actualSampleLen = 0
				seqLen = personImgs[cmcTestInds[i]][0].shape[0]
				if seqLen > sampleSeqLength:
                    			actualSampleLen = sampleSeqLength
                		else:
                    			actualSampleLen = seqLen
				seq_length = actualSampleLen
				seq=personImgs[cmcTestInds[i]][0][0:actualSampleLen,:,:].squeeze().clone()
				#augment each of the images in the sequence
				augSeq=torch.zeros(actualSampleLen,5,56,40)
				#feats_cam_a_mp = []
				for k in xrange(actualSampleLen):
					uu = seq[k,:,:,:].squeeze().clone()
					if doflip == 1:
                        			"""pil_image=to_pillow_image(uu)
						flip_image=pil_image.transpose(Image.FLIP_LEFT_RIGHT) #horizontally flip
            					uu=to_tensor(flip_image)"""
						img_np = uu.numpy()
						img_np = cv2.flip(img_np,0)
						uu = torch.from_numpy(img_np) #to_tensor(img_np)
					uu = uu[:, shiftx: (56 + shiftx), shifty:(40 + shifty)]
					uu = uu - torch.mean(uu)
					augSeq[k,:,:,:] = uu.cuda().clone()
				#pdb.set_trace()
				test_video=Variable(augSeq).cuda()
				#import ipdb;ipdb.set_trace()
				
				output_feature_vec, classifier =net(test_video)
				feats_cam_a[i,:]=output_feature_vec.data.squeeze(0)
			
			feats_cam_b = torch.DoubleTensor(nPersons,128)  
			for i in xrange(nPersons):
				actualSampleLen = 0
                		seqOffset = 0
                		seqLen = personImgs[cmcTestInds[i]][1].shape[0] 
				if seqLen > sampleSeqLength:
                    			actualSampleLen = sampleSeqLength
                    			seqOffset = (seqLen - sampleSeqLength)-1
                		else:
                    			actualSampleLen = seqLen
                    			seqOffset = 0
				seq_length = actualSampleLen
				seq = personImgs[cmcTestInds[i]][1][seqOffset:(seqOffset + actualSampleLen),:,:].squeeze().clone()
				#augment each of the images in the sequence
				augSeq=torch.zeros(actualSampleLen,5,56,40)
				#feats_cam_b_mp = torch.DoubleTensor(actualSampleLen,4096)
				for k in xrange(actualSampleLen):
					uu = seq[k,:,:,:].squeeze().clone()
					if doflip == 1:
                        			"""pil_image=to_pillow_image(uu)
						flip_image=pil_image.transpose(Image.FLIP_LEFT_RIGHT) #horizontally flip
            					uu=to_tensor(flip_image)"""
						img_np = uu.numpy()
						img_np = cv2.flip(img_np,0)
						uu = torch.from_numpy(img_np) #to_tensor(img_np)
					uu = uu[:, shiftx: (56 + shiftx), shifty:(40 + shifty)]
					uu = uu - torch.mean(uu)
					augSeq[k,:,:,:] = uu.cuda().clone()
				test_video=Variable(augSeq).cuda()
				output_feature_vec, classifier=net(test_video)
				feats_cam_b[i,:]=output_feature_vec.data.squeeze(0)

			for i in xrange(nPersons):
				for j in xrange(nPersons):
					fa = feats_cam_a[i,:]
					fb = feats_cam_b[j,:]
					#pdb.set_trace()
					dst = math.sqrt(torch.sum(torch.pow(fa - fb,2)))      #torch.sqrt(torch.sum(torch.pow(fa - fb,2)))
					simMat[i][j] = simMat[i][j] + dst
					if i == j:
						avgSame = avgSame  + dst
                        			avgSameCount = avgSameCount + 1
					else:
                        			avgDiff = avgDiff + dst
                        			avgDiffCount = avgDiffCount + 1

	avgSame = avgSame / avgSameCount
    	avgDiff = avgDiff / avgDiffCount

	cmcInds = torch.DoubleTensor(nPersons)
	cmc=torch.FloatTensor(nPersons).zero_()
        samplingOrder = torch.zeros(nPersons,nPersons)
        for i in xrange(nPersons):
		cmcInds[i] = i
		tmp = simMat[i,:]
		y,o = torch.sort(tmp)

		indx = 0
                tmpIdx = 0
		for j in xrange(nPersons):
            		if o[j] == i:
                		indx = j

		#build the sampling order for the next epoch
		#we want to sample close images i.e. ones confused with this person
			if o[j] is not i:
                		samplingOrder[i][tmpIdx] = o[j]
                		tmpIdx = tmpIdx + 1

		for j in range(indx,nPersons):
            		cmc[j] = cmc[j] + 1

	cmc = (cmc / nPersons) * 100
        cmcString = ''
	#pdb.set_trace()
	for c in xrange(50):
        	if c <= nPersons:
			#pdb.set_trace()
            		cmcString = cmcString+' '+str(int(math.floor(cmc[c])))  #torch.floor

	print(cmcString)
	
	#return cmc,simMat,samplingOrder,avgSame,avgDiff
	return cmc,simMat
				
					
