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

#torch.manual_seed(31415926)
#torch.cuda.manual_seed(31415926)
batch_size = 64
test_batch_size = 128
train_data , test_data = read_CIFAR10(batch_size, test_batch_size)

'''
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
'''
qq = 15

def acquireGradient(netD, dataset1, dataset2, loss_func):
	list1 = dict([])
	list2 = dict([])
	for name, par in netD.named_parameters():
		list1[name] = torch.zeros(par.size()).cuda()
		list2[name] = torch.zeros(par.size()).cuda()
	num1 = 0
	for i,data_batch in enumerate(dataset1):
		feature,label = data_batch
		feature,label = Variable(feature.cuda(), requires_grad = True), Variable(label.cuda())
		
		netD.train()
		netD.zero_grad()
		outputs = netD(feature)
		#error = .0
		#for j in range(outputs.size()[0]):
		#	error -= outputs[j][label.data[j]]
		error = loss_func(outputs, label)
		error.backward()
		for name, par in netD.named_parameters():
			if 'weight' in name:
				list1[name] += par.grad.data
				#if i==qq:
				#	print("%20s %3.4f %3.4f %3.4f %3.4f" %(name, torch.norm(par.data), torch.max(par.data), torch.norm(par.grad.data), torch.max(par.grad.data)))
		
		num1 += 1 
		
	print("*****"*5)
	num2 =0
	for i,data_batch in enumerate(dataset2):
		feature,label = data_batch
		feature,label = Variable(feature.cuda(), requires_grad = True), Variable(label.cuda())
		
		netD.train()
		netD.zero_grad()
		outputs = netD(feature)
		#error = .0
		#for j in range(outputs.size()[0]):
		#	error -= outputs[j][label.data[j]]
		error = loss_func(outputs, label)
		error.backward()
		for name, par in netD.named_parameters():
			if 'weight' in name:
				list2[name] += par.grad.data
				#if i==qq:
				#	print("%20s %3.4f %3.4f %3.4f %3.4f" %(name, torch.norm(par.data), torch.max(par.data), torch.norm(par.grad.data), torch.max(par.grad.data)))
		
		num2 += 1 
	print("*****"*5)

	res1 = 0.0
	res2 = 0.0
	res = 0.0
	#print("===1===")
	#num_term = 0
	for name in list1.keys():
		if 'weight' in name:
			res1 = torch.norm(list1[name]/num1,2)
			res2 = torch.norm(list2[name]/num2,2)
			res = torch.dot(list1[name].view(-1)/(num1*res1), list2[name].view(-1)/(num2*res2))
			print("%20s %2.5f \t %2.5f \t %.4f" %(name, res1, res2, res))
		#print("===2===")
	return 

def acquireAdvGradient(netD, dataset1, dataset2, loss_func):
	list1 = dict([])
	list2 = dict([])
	for name, par in netD.named_parameters():
		list1[name] = torch.zeros(par.size()).cuda()
		list2[name] = torch.zeros(par.size()).cuda()
	num1 = 0
	for i,data_batch in enumerate(dataset1):
		feature,label = data_batch
		netD_cp = copy.deepcopy(netD)
		feature_adv = model_train.advexam_gradient(netD_cp, feature, label, 'sign', coef_FGSM, 1)
		feature_adv, label = Variable(feature_adv), Variable(label.cuda())


		netD.train()
		netD.zero_grad()
		#feature,label = Variable(feature.cuda(), requires_grad = True), Variable(label.cuda())
		outputs = netD(feature_adv)
		#error = .0
		#for j in range(outputs.size()[0]):
		#	error -= outputs[j][label.data[j]]
		error = loss_func(outputs, label)
		error.backward()
		for name, par in netD.named_parameters():
			if 'weight' in name:
				list1[name] += par.grad.data
				#if i==qq:
				#	print("%20s %3.4f %3.4f %3.4f %3.4f" %(name, torch.norm(par.data), torch.max(par.data), torch.norm(par.grad.data), torch.max(par.grad.data)))
		num1 += 1 
		
	print("*****"*5)
	num2 =0
	for i,data_batch in enumerate(dataset2):
		feature,label = data_batch
		netD_cp = copy.deepcopy(netD)
		feature_adv = model_train.advexam_gradient(netD_cp, feature, label, 'sign', coef_FGSM, 1)
		feature_adv, label = Variable(feature_adv), Variable(label.cuda())


		netD.train()
		netD.zero_grad()
		#feature,label = Variable(feature.cuda(), requires_grad = True), Variable(label.cuda())
		outputs = netD(feature_adv)
		#error = .0
		#for j in range(outputs.size()[0]):
		#	error -= outputs[j][label.data[j]]
		error = loss_func(outputs, label)
		error.backward()
		for name, par in netD.named_parameters():
			if 'weight' in name:
				list2[name] += par.grad.data
				#if i==qq:
				#	print("%20s %3.4f %3.4f %3.4f %3.4f" %(name, torch.norm(par.data), torch.max(par.data), torch.norm(par.grad.data), torch.max(par.grad.data)))
		num2 += 1 

	print("*****"*5)
	res1 = 0.0
	res2 = 0.0
	res = 0.0
	#print("===1===")
	#num_term = 0
	for name in list1.keys():
		if 'weight' in name:
			res1 = (list1[name]/num1).norm(p=2)
			res2 = (list2[name]/num2).norm(p=2)
			res = torch.dot(list1[name].view(-1)/(num1*res1), list2[name].view(-1)/(num2*res2))
			print("%20s %2.5f \t %2.5f \t %.4f" %(name, res1, res2, res))
		#print("===2===")
	return 

netD = _netD_cifar10()
netD.cuda()
#netD.load_state_dict(torch.load('netD.pkl'))
#print('Test accuracy of netD: %.3f'%(TestAcc_dataloader(netD,test_data)))
loss_func = nn.CrossEntropyLoss()
#netD_cp = copy.deepcopy(netD)
netD.train()
print("===="*5)
acquireGradient(netD, train_data, test_data, loss_func)
print("===="*5)
acquireAdvGradient(netD, train_data, test_data, loss_func)

netD = _netD_cifar10()
netD.cuda()
optimizerD = optim.SGD(netD.parameters(), lr=0.01, weight_decay = 0.0002)


for epoch in range(epoch_num):
	running_loss_D = .0
	running_acc_D = .0
	for i, data_batch in enumerate(train_data):
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
		running_acc_D += accu(outputs, label)/batch_size

		if i%150==149:
			print('[%d/%d][%d/%d] Adv perf: %.4f / %.4f'
				% (epoch, epoch_num, i, len(train_data),
					running_loss_D.data.cpu().numpy()[0]/150, running_acc_D/150))
			running_loss_D = .0
			running_acc_D = .0
		if epoch%5==2 and i%500==0:
			vutils.save_image(feature_adv.data, './adv_image/adv_image_epoch_%03d_%03d.png' %(epoch,i), normalize = True)
	if epoch in {10,20,25,30}:
				optimizerD.param_groups[0]['lr'] /= 2.0
	if epoch % 5 == 0:
		dataset_adv = torch.load('./adv_exam/adv_gradient_FGSM_step1.pt')
		netD.eval()
		test_acc = TestAcc_dataloader(netD, test_data)
		test_adv_acc = TestAcc_tensor(netD, dataset_adv)
		print('[%d/%d]Test accu: %.3f' %(epoch, epoch_num, test_acc) )
		print('[%d/%d]Test ADV accu: %.3f' %(epoch, epoch_num, test_adv_acc) )
		print('[%d/%d]White-box Attack train accuracy: %.3f' %(epoch, epoch_num,TestAdvAcc_dataloader(netD, train_data, 'sign', 0.03)))
		print('[%d/%d]White-box Attack test accuracy: %.3f' %(epoch, epoch_num,TestAdvAcc_dataloader(netD, test_data, 'sign', 0.03)))
		
		print("===="*5)
		acquireGradient(netD, train_data, test_data, loss_func)
		print("===="*5)
		acquireAdvGradient(netD, train_data, test_data, loss_func)
		
		#print(acquireGradient(netD_cp, train_data, loss_func), acquireGradient(netD_cp, test_data, loss_func),acquireGradient(netD_cp, inter_data, loss_func))
			
	


