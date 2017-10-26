from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.models as models

from nets.classifiers import _netD_cifar10,_netG_cifar10
from constants import *
from utils.utils import accu,TestAcc_dataloader, TestAcc_tensor, TestAdvAcc_dataloader
import models.model_train as model_train
from dataProcess.read_data import read_CIFAR10
import glob
import copy
import numpy as np
'''
This exp is to test the variance and the bias of NN. The true function values 
are defined the pre-softmax layer of a NN which minimize the training error. 
The distribution of (X,\hat{y}) is fixed to be the empirical data.

Estimation of the bias and the variance::
We train a square loss of the same/different structure of the networks. 
We use sample with replacement from the empirical data to create a new
training set. Then compute the 'true' loss of the learned network.

This precedure is repeated multiple times to estimate the variance of the 
model, and the 'true' bias of the model.

'''

batch_size = 128
test_batch_size = 128
train_data , valid_data, test_data = read_CIFAR10(batch_size, test_batch_size, 0.2)
attack_method = model_train.advexam_gradient


## Create dataset
netD = _netD_cifar10()
netD.cuda()
#netD.load_state_dict(torch.load('netD.pkl'))
loss_func = nn.CrossEntropyLoss()
optimizerD = optim.SGD(netD.parameters(), lr=0.01, weight_decay = 0.0002)
epoch_num = 70

for epoch in range(epoch_num):
	running_loss_D = .0
	running_acc_D = .0

	for i, data_batch in enumerate(train_data):
		image_batch,label_batch = data_batch
		image_batch, label_batch = Variable(image_batch.cuda()), Variable(label_batch.cuda())
		
		netD.train()
		netD.zero_grad()
		outputs = netD(image_batch)
		error = loss_func(outputs, label_batch)
		error.backward()
		optimizerD.step()

		running_acc_D += accu(outputs,label_batch)/batch_size 
		running_loss_D += error

		if i%150==149:
			print('[%d/%d][%d/%d] Adv perf: %.4f / %.4f'
				% (epoch, epoch_num, i, len(train_data),
					running_loss_D.data.cpu().numpy()[0]/150, running_acc_D/150))
			running_loss_D = .0
			running_acc_D = .0
	if epoch%5 ==0:
		print("===="*5)
		netD.eval()
		train_acc = TestAcc_dataloader(netD, train_data)
		valid_acc = TestAcc_dataloader(netD, valid_data)
		#test_adv_acc = TestAcc_tensor(netD, dataset_adv)
		print('[%d/%d]Train accu: %.3f' %(epoch, epoch_num, train_acc) )
		print('[%d/%d]Valid  accu: %.3f' %(epoch, epoch_num, valid_acc))

netD.eval()
test_acc = TestAcc_dataloader(netD, test_data)
print('[%d/%d]Test accu: %.3f' %(epoch, epoch_num, test_acc) )

#Create dataset
image_tr = []
label_tr = []
iamge_val = []
label_val = []
image_t = []
label_t = []

for i,data_batch in enumerate(train_data):
	image_batch,label_batch = data_batch
	image_batch_var, label_batch_var = Variable(image_batch.cuda()), Variable(label_batch.cuda())
	outputs = netD(image_batch_var)
	image_tr.append(image_batch.cuda())
	label_tr.append(outputs.data)

for i,data_batch in enumerate(valid_data):
	image_batch,label_batch = data_batch
	image_batch_var, label_batch_var = Variable(image_batch.cuda()), Variable(label_batch.cuda())
	outputs = netD(image_batch_var)
	image_val.append(image_batch.cuda())
	label_val.append(outputs.data)

for i,data_batch in enumerate(test_data):
	image_batch,label_batch = data_batch
	image_batch_var, label_batch_var = Variable(image_batch.cuda()), Variable(label_batch.cuda())
	outputs = netD(image_batch_var)
	image_t.append(image_batch.cuda())
	label_t.append(outputs.data)

image_tr = torch.cat(image_tr+image_val, 0)
label_tr = torch.cat(label_tr+label_val, 0)
#image_val= torch.cat(image_val, 0)
#label_val = torch.cat(label_val, 0)
image_t = torch.cat(image_t, 0)
label_t = torch.cat(label_t, 0)

train_dataset = torch.utils.data.TensorDataset(image_tr, label_tr)
test_dataset = torch.utils.data.TensorDataset(image_t, label_t)

res = []
num_train = len(train_dataset)
split = int(np.floor(0.2 * num_train))

for i in range(num_trial):
	netD = _netD_cifar10()
	netD.cuda()
	loss_func = nn.MSELoss()
	optimizerD = optim.SGD(netD.parameters(), lr=0.01, weight_decay = 0.0002)
	epoch_num = 25

	indices = list(range(num_train))
	np.random.shuffle(indices)
	train_idx, valid_idx = indices[split:], indices[:split]
	train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
	valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)

	train_data = torch.utils.data.DataLoader(train_dataset,
		batch_size=batch_size, sampler=train_sampler, drop_last = True)
	valid_data = torch.utils.data.DataLoader(train_dataset,
		batch_size=batch_size, sampler=valid_sampler, drop_last = True)

	for epoch in range(epoch_num):
		for j,data_batch in enumerate(train_data)
			netD.zero_grad()
			feature,label = data_batch
			feature_v, label_v = Variable(feature.cuda()), Variable(label.cuda())
			outputs_v = netD(feature_v)
			loss = loss_func(outputs_v, label_v)
			loss.backward()
			optimizerD.step()
			running_loss += loss.data[0]
			running_acc += accu(outputs_v, label_v)/batch_size
			if i % 100 ==99:
				print('[%d, %5d] loss: %.3f accuracy: %.3f' %(epoch + 1, i + 1,
		 			running_loss / 100, running_acc/100))
				running_loss = .0
				running_acc = .0
			print('Test accuracy of netD: %.3f'%(TestAcc_dataloader(netD,test_data)))
		if epoch in {5,10,20,30}:
			optimizerD.param_groups[0]['lr'] /= 2.0












