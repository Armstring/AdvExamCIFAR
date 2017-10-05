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

from nets.classifiers import _netD_mnist,_netG_mnist
from constants import *
from utils.utils import accu,TestAcc_dataloader, TestAcc_tensor
import models.model_train as model_train
from dataProcess.read_data import read_MNIST
import glob
import copy

#torch.manual_seed(31415926)
#torch.cuda.manual_seed(31415926)
batch_size = 64
test_batch_size = 1000
train_data , test_data = read_MNIST(batch_size, test_batch_size)

feature_inter_list = []
label_list = []
feature_pre = torch.zeros(batch_size,nc_netD,image_shape[0],image_shape[1])
for i, data_batch in enumerate(train_data):
	feature,label = data_batch
	feature_inter = (feature+feature_pre)/2.0
	feature_inter_list.append(feature_inter)
	feature_pre = feature
	label_list.append(label)
feature_inter_set = torch.cat(feature_inter_list, 0)
labelset = torch.cat(label_list, 0)
torch.save((feature_inter_set, labelset), 'inter.pt')

inter_data = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(feature_inter_set, labelset),
    batch_size=64, shuffle=True, drop_last = False)

netD = _netD_mnist()
netD.cuda()
netD.load_state_dict(torch.load('netD.pkl'))
print('Test accuracy of netD: %.3f'%(TestAcc_dataloader(netD,test_data)))
optimizerD = optim.Adam(netD.parameters(), lr=lr_D, betas=(0.9, 0.999))
loss_func = nn.CrossEntropyLoss()

def acquireGradient(netD, dataset1, dataset2, loss_func):
	res1 = torch.zeros(nc_netD,image_size).cuda()
	num1 = 0
	for i,data_batch in enumerate(dataset1):
		feature,label = data_batch
		feature,label = Variable(feature.cuda(), requires_grad = True), Variable(label.cuda())
		outputs = netD(feature)
		error = .0
		for j in range(outputs.size()[0]):
			error -= outputs[j][label.data[j]]
		#error = loss_func(outputs, label)
		error.backward()
		res1 += torch.sum(feature.grad.data, 0)
		num1 += feature.size()[0]

	res2 = torch.zeros(1,28*28).cuda()
	num2 = 0
	for i,data_batch in enumerate(dataset2):
		feature,label = data_batch
		feature,label = Variable(feature.cuda(), requires_grad = True), Variable(label.cuda())
		outputs = netD(feature)
		error = .0
		for j in range(outputs.size()[0]):
			error -= outputs[j][label.data[j]]
		#error = loss_func(outputs, label)
		error.backward()
		res2 += torch.sum(feature.grad.data, 0)
		num2 += feature.size()[0]

	return (torch.norm(1.0*res1/num1), torch.norm(1.0*res2/num2), torch.norm(1.0*res1/num1 - 1.0*res2/num2))


netD_cp = copy.deepcopy(netD)
print(acquireGradient(netD_cp, train_data, test_data, loss_func))

for epoch in range(epoch_num):
	running_loss_D = .0
	running_acc_D = .0
	for i, data_batch in enumerate(train_data):
		#netD_cp = copy.deepcopy(netD)
		#gradient_tr[epoch][i] = acquireGradient(netD_cp, train_data, loss_func)
		#gradient_t[epoch][i] = acquireGradient(netD_cp, test_data, loss_func)
		#gradient_inter[epoch][i] = acquireGradient(netD_cp, inter_data, loss_func)

		netD.train()

		netD.zero_grad()
		feature,label = data_batch
		netD_cp = copy.deepcopy(netD)
		feature_adv = model_train.advexam_gradient(netD_cp, feature, label, 'sign', coef_FGSM, 1)
		feature_adv, label = Variable(feature_adv), Variable(label.cuda())
				
		outputs = netD(feature_adv.detach())
		errorD_real = loss_func(outputs, label)
		errorD_real.backward()
		optimizerD.step()

		running_loss_D += errorD_real
		running_acc_D += accu(outputs, label)

		if i%50==49:
			print('[%d/%d][%d/%d] Adv perf: %.4f / %.4f'
				% (epoch, epoch_num, i, len(train_data),
					running_loss_D.data.cpu().numpy()[0]/50, running_acc_D/50))
			running_loss_D = .0
			running_acc_D = .0
		if epoch%5==2 and i%500==0:
			vutils.save_image(feature_adv.data, './adv_image/adv_image_epoch_%03d_%03d.png' %(epoch,i), normalize = True)
	if epoch % 3 == 0:
		dataset_adv = torch.load('./adv_exam/adv_gradient_FGSM_step15.pt')
		netD.eval()
		test_acc = TestAcc_dataloader(netD, test_data)
		test_adv_acc = TestAcc_tensor(netD, dataset_adv)
		print('[%d/%d]Test accu: %.3f' %(epoch, epoch_num, test_acc) )
		print('[%d/%d]Test ADV accu: %.3f' %(epoch, epoch_num, test_adv_acc) )
		netD.train()
		netD_cp = copy.deepcopy(netD)
		print(acquireGradient(netD_cp, train_data, test_data, loss_func))
		#print(acquireGradient(netD_cp, train_data, loss_func), acquireGradient(netD_cp, test_data, loss_func),acquireGradient(netD_cp, inter_data, loss_func))
			
	


