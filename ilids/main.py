'''Video based person re-identification'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from utils import progress_bar
from torch.autograd import Variable
from prepareDataset import prepareDataset
from datasetUtils import partitionDataset,feature_extraction_using_vgg
from build_model import build_model
from train import train_test_Sequence
from test import computeCMC_MeanPool_RNN
import ipdb

#58 e-4 after 400 or 600

parser = argparse.ArgumentParser(description='PyTorch Person Re-Identification from Video')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--dataset', default=1, type=int, help='1 -  ilids, 2 - prid')
parser.add_argument('--nEpochs', default=1500, type=int, help='number of training epochs')
parser.add_argument('--gpu', default=2, type=int, help='Which gpu should use')
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--sampleSeqLength', default=16, type=int, help='length of sequence to train network')
parser.add_argument('--momentum', default=0.9,type=float,help='momentum')
parser.add_argument('--samplingEpochs', default=100, type=int, help='how often to compute the CMC curve')
parser.add_argument('-disableOpticalFlow',default='false', type= str, help='use optical flow features or not')

"""parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch"""

args = parser.parse_args()

print(args)

print('=================== FCN 32 ========================')
#set cuda
"""os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)"""

torch.cuda.set_device(device=args.gpu)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic=True
import random;random.seed(0)

# path of the dataset
if args.dataset==1:
    seqRootRGB = '../3d-fcn/data/i-LIDS-VID/i-LIDS-VID/sequences/'
    seqRootOF = '../3d-fcn/data/i-LIDS-VID-OF-HVP/sequences/'
else:
    seqRootRGB = '../3d-fcn/data/PRID2011/multi_shot/'
    seqRootOF = './3d-fcn/data/PRID2011-OF-HVP/multi_shot/'

# Data
print('loading Dataset - ',seqRootRGB,seqRootOF)
dataset = prepareDataset(seqRootRGB, seqRootOF, '.png', args.dataset,args.disableOpticalFlow)
print('dataset loaded')

print('randomizing test/training split')
trainInds,testInds = partitionDataset(len(dataset),0.5)

#load FCN network
fcn_attention_model=build_model()

"""
#feature extraction using vgg
feature_net = torchvision.models.vgg19(pretrained='imagenet') #pre-train vgg model

# convert all the layers to list and remove the last three
features = list(feature_net.classifier.children())[:-3]

## Add the last layer based on the num of classes in our dataset
#features.extend([nn.Linear(num_ftrs, n_class)])

## convert it into container and add it to our model class.
feature_net.classifier = nn.Sequential(*features)
feature_net.eval()
feature_net.cuda()

#feature extraction from each video sequence
vgg_extracted_feature=feature_extraction_using_vgg(feature_net,dataset)

#delete feature extraction network
#del feature_net
"""
#train the model
model=train_test_Sequence(fcn_attention_model, dataset, trainInds, testInds, args.nEpochs, args.sampleSeqLength, args.lr,args.momentum, args.samplingEpochs)

#save trained model
torch.save(model, 'trained_model/model.pt')

#load trained model
#model = torch.load('trained_model/model.pt')

model.eval()
nTestImages = [2,4,8,16,32,64,128]

#need to save model

for n in xrange(len(nTestImages)):
	print('test multiple images '+str(nTestImages[n]))
	#default method of computing CMC curve
	computeCMC_MeanPool_RNN(dataset,testInds,model,nTestImages[n])


