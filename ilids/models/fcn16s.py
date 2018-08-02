import os.path as osp

# import fcn						# changed this
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import pdb


class FCN16s(nn.Module):

    def __init__(self, n_class=1):
        super(FCN16s, self).__init__()

	"""
	#pre-train vgg model
	self.image_feature=torchvision.models.vgg19(pretrained='imagenet') 
	
	#convert all the layers to list and remove the last three
	features = list(self.image_feature.classifier.children())[:-3]
	self.image_feature.classifier = nn.Sequential(*features)"""

	padDim = 4
	nFilters = [16,32,32]
	filtsize = [5,5,5]
    	poolsize = [2,2,2]
    	stepSize = [2,2,2]

	ninputChannels = 5

 	self.padding_1 = nn.ZeroPad2d(padDim)
	self.cnn_conv1 = nn.Conv2d(ninputChannels, nFilters[0], (filtsize[0], filtsize[0]), (1, 1))
	self.tanh1 = nn.Tanh()
	self.maxpool1 = nn.MaxPool2d((poolsize[0],poolsize[0]),(stepSize[0],stepSize[0])) 

	ninputChannels = nFilters[0]
	self.padding_2 = nn.ZeroPad2d(padDim)
	self.cnn_conv2 = nn.Conv2d(ninputChannels, nFilters[1], (filtsize[1], filtsize[1]), (1, 1))
	self.tanh2 = nn.Tanh()
	self.maxpool2 = nn.MaxPool2d((poolsize[1],poolsize[1]),(stepSize[1],stepSize[1])) 

	ninputChannels = nFilters[1]
	self.padding_3 = nn.ZeroPad2d(padDim)
	self.cnn_conv3 = nn.Conv2d(ninputChannels, nFilters[2], (filtsize[2], filtsize[2]), (1, 1))
	self.tanh3 = nn.Tanh()
	self.maxpool3 = nn.MaxPool2d((poolsize[2],poolsize[2]),(stepSize[2],stepSize[2])) 
	
	nFullyConnected = nFilters[2]*10*8 

	self.cnn_drop = nn.Dropout2d(p=0.6)
	self.linear = nn.Linear(nFullyConnected,128)

        # conv1
        self.conv1_1 = nn.Conv2d(128, 64, (1,3), padding=(0,100))
	self.bnorm1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.Tanh()
        self.conv1_2 = nn.Conv2d(64, 64, (1,3), padding=(0,1))
        self.bnorm1_2 = nn.BatchNorm2d(64)
        self.relu1_2 = nn.Tanh()
        self.pool1 = nn.MaxPool2d((1,2), stride=(1,2), ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, (1,3), padding=(0,1))
        self.bnorm2_1 = nn.BatchNorm2d(128)
        self.relu2_1 = nn.Tanh()
        self.conv2_2 = nn.Conv2d(128, 128, (1,3), padding=(0,1))
        self.bnorm2_2 = nn.BatchNorm2d(128)
        self.relu2_2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d((1,2), stride=(1,2), ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, (1,3), padding=(0,1))
        self.bnorm3_1 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.Tanh()
        self.conv3_2 = nn.Conv2d(256, 256, (1,3), padding=(0,1))
        self.bnorm3_2 = nn.BatchNorm2d(256)
        self.relu3_2 = nn.Tanh()
        self.conv3_3 = nn.Conv2d(256, 256, (1,3), padding=(0,1))
        self.bnorm3_3 = nn.BatchNorm2d(256)
        self.relu3_3 = nn.Tanh()
        self.pool3 = nn.MaxPool2d((1,2), stride=(1,2), ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, (1,3), padding=(0,1))
        self.bnorm4_1 = nn.BatchNorm2d(512)
        self.relu4_1 = nn.Tanh()
        self.conv4_2 = nn.Conv2d(512, 512, (1,3), padding=(0,1))
        self.bnorm4_2 = nn.BatchNorm2d(512)
        self.relu4_2 = nn.Tanh()
        self.conv4_3 = nn.Conv2d(512, 512, (1,3), padding=(0,1))
        self.bnorm4_3 = nn.BatchNorm2d(512)
        self.relu4_3 = nn.Tanh()
        self.pool4 = nn.MaxPool2d((1,2), stride=(1,2), ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, (1,3), padding=(0,1))
        self.bnorm5_1 = nn.BatchNorm2d(512)
        self.relu5_1 = nn.Tanh()
        self.conv5_2 = nn.Conv2d(512, 512, (1,3), padding=(0,1))
        self.bnorm5_2 = nn.BatchNorm2d(512)
        self.relu5_2 = nn.Tanh()
        self.conv5_3 = nn.Conv2d(512, 512, (1,3), padding=(0,1))
        self.bnorm5_3 = nn.BatchNorm2d(512)
        self.relu5_3 = nn.Tanh()
        self.pool5 = nn.MaxPool2d((1,2), stride=(1,2), ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, (1,7))
        self.relu6 = nn.Tanh()
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, (1,1))
        self.relu7 = nn.Tanh()
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_class, (1,1))
        self.score_fr_bn = nn.BatchNorm2d(n_class)
        self.score_pool4 = nn.Conv2d(512, n_class, (1,1))
        self.score_pool4_bn = nn.BatchNorm2d(n_class)

        self.upscore2 = nn.ConvTranspose2d(
            n_class, n_class, (1,4), stride=(1,2), bias=False)
        self.upscore16 = nn.ConvTranspose2d(
            n_class, n_class, (1,32), stride=(1,16), bias=False)

	self.Sigmoid=nn.Sigmoid()

	self.classifierLayer = nn.Linear(128,150)

	self.logsoftmax=nn.LogSoftmax()

        #self._initialize_weights()


    def forward(self, x):

	"""x_feature=self.image_feature(x)"""
	#pdb.set_trace()
	cnn = self.padding_1(x)
	cnn = self.cnn_conv1(cnn)
	cnn = self.tanh1(cnn)
	cnn = self.maxpool1(cnn)

	cnn=self.padding_2(cnn)
	cnn = self.cnn_conv2(cnn)
	cnn = self.tanh2(cnn)
	cnn = self.maxpool2(cnn)

	cnn=self.padding_3(cnn)
	cnn = self.cnn_conv3(cnn)
	cnn = self.tanh3(cnn)
	cnn = self.maxpool3(cnn)
	
	nFullyConnected = 32*10*8  #nFilters[1]
	#pdb.set_trace()
	cnn = cnn.view(-1,nFullyConnected)
	cnn = self.cnn_drop(cnn)
	cnn = self.linear(cnn)
	
	feature_fcn = cnn.transpose(0,1).unsqueeze(0).unsqueeze(2)
	
        h = feature_fcn
        h = self.relu1_1(self.bnorm1_1(self.conv1_1(h)))
        h = self.relu1_2(self.bnorm1_2(self.conv1_2(h)))
        h = self.pool1(h)
       
        h = self.relu2_1(self.bnorm2_1(self.conv2_1(h)))
        h = self.relu2_2(self.bnorm2_2(self.conv2_2(h)))
        h = self.pool2(h)
      
        h = self.relu3_1(self.bnorm3_1(self.conv3_1(h)))
        h = self.relu3_2(self.bnorm3_2(self.conv3_2(h)))
        h = self.relu3_3(self.bnorm3_3(self.conv3_3(h)))
        h = self.pool3(h)
      
        h = self.relu4_1(self.bnorm4_1(self.conv4_1(h)))
        h = self.relu4_2(self.bnorm4_2(self.conv4_2(h)))
        h = self.relu4_3(self.bnorm4_3(self.conv4_3(h)))
        h = self.pool4(h)
        pool4 = h  # 1/16
       
        h = self.relu5_1(self.bnorm5_1(self.conv5_1(h)))
        h = self.relu5_2(self.bnorm5_2(self.conv5_2(h)))
        h = self.relu5_3(self.bnorm5_3(self.conv5_3(h)))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

	#if not self.training:	
	#import ipdb; ipdb.set_trace()
	
        h = self.score_fr(h) #self.score_fr_bn(self.score_fr(h))
        h = self.upscore2(h)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4) #self.score_pool4_bn(self.score_pool4(pool4))
        
        h = h[:, :, :, 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c

        h = self.upscore16(h)
        h = h[:, :, :, 27:27 + feature_fcn.size()[3]].contiguous()		#always check here

        attention_score=h.squeeze(0).squeeze(0).transpose(0,1)
	attention_score=self.Sigmoid(attention_score)

	"""attention_score=h.squeeze(0).squeeze(0)
	attention_score=self.softmax(attention_score).transpose(0,1)"""

	cnn_transpose=cnn.transpose(0,1)
	
	attention_feature = torch.mm(cnn_transpose,attention_score)
	attention_feature = attention_feature.transpose(0,1)

	combine_feature = torch.cat((cnn,attention_feature),0)

	combine_feature = torch.mean(combine_feature,0).unsqueeze(0)

	combine_feature = F.normalize(combine_feature, p=2, dim=1)

	classifier = self.classifierLayer(combine_feature)
	logsoftmax = self.logsoftmax(classifier)

        return combine_feature,logsoftmax

