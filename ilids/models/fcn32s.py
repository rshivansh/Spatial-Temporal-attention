import os.path as osp

# import fcn							# changed this
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


import ipdb
class FCN32s(nn.Module):

    def __init__(self, n_class=1):
        super(FCN32s, self).__init__()

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
	
	self.sp_conv_attention = nn.Conv2d(32, 3, (5,5), padding=(2,2))
	self.sp_conv = nn.Conv2d(32, 32, (5,5), padding=(2,2))
	self.sp_sigmoid = nn.Sigmoid()

	self.cnn_drop = nn.Dropout2d(p=0.6)
	self.linear = nn.Linear(nFullyConnected,128)

	self.sp_cnn_drop = nn.Dropout2d(p=0.6)
	self.sp_linear = nn.Linear(nFullyConnected,128)

        # fc6
        self.fc6 = nn.Conv2d(128, 1, (1,1))
        self.drop6 = nn.Dropout2d()

    	self.Sigmoid=nn.Sigmoid()
	self.classifierLayer = nn.Linear(128,150)
	self.logsoftmax=nn.LogSoftmax()

    def forward(self, x):

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
	
	sp_cnn=cnn

	sp_attention = self.sp_conv_attention(sp_cnn)

	sp_attention1=sp_attention[:,0].unsqueeze(1)
	sp_attention2=sp_attention[:,1].unsqueeze(1)
	sp_attention3=sp_attention[:,2].unsqueeze(1)

	sp_sig1 = self.sp_sigmoid(sp_attention1)
	sp_sig2 = self.sp_sigmoid(sp_attention2)
	sp_sig3 = self.sp_sigmoid(sp_attention3)

	#sp_sig = self.sp_sigmoid(sp_attention)
	
	nFullyConnected = 32*10*8  #nFilters[1]

	cnn = cnn.view(-1,nFullyConnected)
	cnn = self.cnn_drop(cnn)
	cnn = self.linear(cnn)

	feature_fcn = cnn.transpose(0,1).unsqueeze(0).unsqueeze(2)   #   # (1L, 128L, 1L, 16L)
	
        h = feature_fcn
        h = self.fc6(h)
        h = self.drop6(h) # 1 128 1 16

        attention_score =h
	attention_score=self.Sigmoid(attention_score)
	attention_score=attention_score.squeeze(0)
	attention_score=attention_score.squeeze(1)
	attention_score=attention_score.transpose(0,1)
	cnn_transpose=cnn.transpose(0,1)
	ipdb.set_trace()
	attention_score=torch.mm(cnn_transpose,attention_score)
	attention_score=attention_score.transpose(0,1)
	spatial_attention1 = torch.mul( sp_cnn, sp_sig1)
	spatial_attention2 = torch.mul( sp_cnn, sp_sig2)
	spatial_attention3 = torch.mul( sp_cnn, sp_sig3)

	spatial_attention1 = spatial_attention1.view(-1,nFullyConnected)
	spatial_attention2 = spatial_attention2.view(-1,nFullyConnected)
	spatial_attention3 = spatial_attention3.view(-1,nFullyConnected)

	#spatial_attention = torch.mul( sp_cnn, sp_sig)
	#spatial_attention = spatial_attention.view(-1,nFullyConnected)
	#spatial_attention = self.sp_cnn_drop(spatial_attention)
	#spatial_attention = self.sp_linear(spatial_attention)

	spatial_attention1 = self.sp_cnn_drop(spatial_attention1)
	spatial_attention1 = self.sp_linear(spatial_attention1)

	spatial_attention2 = self.sp_cnn_drop(spatial_attention2)
	spatial_attention2 = self.sp_linear(spatial_attention2)

	spatial_attention3 = self.sp_cnn_drop(spatial_attention3)
	spatial_attention3 = self.sp_linear(spatial_attention3)

	#combine_feature = torch.mul(spatial_attention,attention_score)
	#combine_feature = torch.cat((cnn,combine_feature),0)

	#combine_feature = torch.mean(combine_feature,0).unsqueeze(0)

	#combine_feature= F.normalize(combine_feature, p=2, dim=1)
	combine_feature1 = torch.add(spatial_attention1,attention_score)
	combine_feature2 = torch.add(spatial_attention2,attention_score)
	combine_feature3 = torch.add(spatial_attention3,attention_score)

	combine_feature1 = torch.cat((cnn,combine_feature1),0)
	combine_feature2 = torch.cat((cnn,combine_feature2),0)
	combine_feature3 = torch.cat((cnn,combine_feature3),0)

	combine_feature1 = torch.mean(combine_feature1,0).unsqueeze(0)
	combine_feature2 = torch.mean(combine_feature2,0).unsqueeze(0)
	combine_feature3 = torch.mean(combine_feature3,0).unsqueeze(0)

	combine_feature1= F.normalize(combine_feature1, p=2, dim=1)
	combine_feature2= F.normalize(combine_feature2, p=2, dim=1)
	combine_feature3= F.normalize(combine_feature3, p=2, dim=1)

	combine_feature=torch.add(combine_feature1,combine_feature2)
	combine_feature=torch.add(combine_feature,combine_feature3)
	classifier = self.classifierLayer(combine_feature)
	logsoftmax = self.logsoftmax(classifier)

        return combine_feature,logsoftmax


