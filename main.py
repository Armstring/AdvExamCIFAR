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
train_data, valid_data, test_data = read_CIFAR10(batch_size, test_batch_size, 0.2, True)


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
attack_method = model_train.advexam_gradient
epoch_num = 60
max_mag = 0.01
num_iter = 7
coef_FGSM = 0.0025


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
		num1 += 1 
		
	#print("*****"*5)
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
		num2 += 1 
	#print("*****"*5)

	#res1 = 0.0
	#res2 = 0.0
	#res = 0.0
	#print("===1===")
	#num_term = 0
	#for name in list1.keys():
		#if 'weight' in name:
			#res1 = torch.norm(list1[name]/num1,2)
			#res2 = torch.norm(list2[name]/num2,2)
			#res = torch.dot(list1[name].view(-1)/(num1*res1), list2[name].view(-1)/(num2*res2))
			#print("%20s %2.5f \t %2.5f \t %.4f" %(name, res1, res2, res))
		#print("===2===")
#	return 

#def acquireAdvGradient(netD, dataset1, dataset2, loss_func):
	list3 = dict([])
	list4 = dict([])
	for name, par in netD.named_parameters():
		list3[name] = torch.zeros(par.size()).cuda()
		list4[name] = torch.zeros(par.size()).cuda()
	num3 = 0
	for i,data_batch in enumerate(dataset1):
		feature,label = data_batch
		netD_cp = copy.deepcopy(netD)
		feature_adv = attack_method(netD_cp, feature, label, 'sign', coef_FGSM, num_iter,max_mag)
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
				list3[name] += par.grad.data
		num3 += 1 
		
	#print("*****"*5)
	num4 =0
	for i,data_batch in enumerate(dataset2):
		feature,label = data_batch
		netD_cp = copy.deepcopy(netD)
		feature_adv = attack_method(netD_cp, feature, label, 'sign', coef_FGSM, num_iter,max_mag)
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
				list4[name] += par.grad.data
		num4 += 1 

	print("*****"*5)
	res1 = 0.0
	res2 = 0.0
	res3 = 0.0
	res4 = 0.0
	#res = 0.0
	for name in list1.keys():
		if 'weight' in name:
			res1 = (list1[name]/num1).norm(p=2)
			res2 = (list2[name]/num2).norm(p=2)
			res3 = (list3[name]/num3).norm(p=2)
			res4 = (list4[name]/num4).norm(p=2)
			res_cor1 = torch.dot(list1[name].view(-1)/(num1*res1), list3[name].view(-1)/(num3*res3))

			res_cor2 = torch.dot(list2[name].view(-1)/(num2*res2), list3[name].view(-1)/(num3*res3))
			
			res_cor3 = torch.dot(list4[name].view(-1)/(num4*res4), list3[name].view(-1)/(num3*res3))
			
			print("%20s %2.5f \t %2.5f \t %.4f" %(name, res_cor1, res_cor2, res_cor3))
		#print("===2===")
	return 

def acquireInputGradient(netD, dataset1, dataset2, loss_func):
	netD.eval()
	res1 = 0.0
	res1_error = 0.0
	num1 = 0
	for i, data_batch in enumerate(dataset1):
		feature, label = data_batch
		#gaussnoise = torch.zeros(feature.size()).cuda()
		#gaussnoise.normal_()
		#feature = feature + 0.02* gaussnoise
		#netD_cp = copy.deepcopy(netD)
		perturb = torch.zeros(feature.size()).cuda()
		feature, label = Variable(feature.cuda()), Variable(label.cuda())
		perturb = Variable(perturb, requires_grad = True)
		feature_adv = feature + perturb
		outputs = netD(feature_adv)
		_, pred = torch.max(outputs, 1)
		error = loss_func(outputs, label)

		error.backward()
		res_temp = 0.0
		for j,image in enumerate(feature):
			if pred.data[j]==label.data[j]:
				res_temp  += torch.sum(torch.abs(perturb.grad[j].data))
		res1 += res_temp/feature.size()[0]
		res1_error += error.data[0]
		num1 +=1
	res2 = 0.0
	res2_error = 0.0
	num2 = 0
	for i, data_batch in enumerate(dataset2):
		feature, label = data_batch
		#gaussnoise = torch.zeros(feature.size()).cuda()
		#gaussnoise.normal_()
		#feature = feature + 0.02* gaussnoise
		#netD_cp = copy.deepcopy(netD)
		perturb = torch.zeros(feature.size()).cuda()
		feature, label = Variable(feature.cuda()), Variable(label.cuda())
		perturb = Variable(perturb, requires_grad = True)
		feature_adv = feature + perturb
		outputs = netD(feature_adv)
		_, pred = torch.max(outputs, 1)
		error = loss_func(outputs, label)

		error.backward()
		res_temp = 0.0
		for j,image in enumerate(feature):
			if True:#pred.data[j]==label.data[j]:
				res_temp  += torch.sum(torch.abs(perturb.grad[j].data))
		res2 += res_temp/feature.size()[0]
		res2_error += error.data[0]
		num2 +=1
	print("Valid Loss: %3.5f; Test Loss: %3.5f" %(res1_error/num1, res2_error/num2))
	return


netD = _netD_cifar10()
netD.cuda()
#netD.load_state_dict(torch.load('netD.pkl'))
#print('Test accuracy of netD: %.3f'%(TestAcc_dataloader(netD,test_data)))
loss_func = nn.CrossEntropyLoss()

#optimizerD = optim.SGD(netD.parameters(), lr=0.01, weight_decay = 0.0002)
optimizerD = optim.Adam(netD.parameters(), lr=lr_D, betas=(0.5, 0.999), weight_decay = 0.0002)


for epoch in range(epoch_num):
	running_loss_D = .0
	running_acc_D = .0
	mag = 0.0
	for i, data_batch in enumerate(train_data):
		netD.train()

		netD.zero_grad()
		feature,label = data_batch
		netD_cp = copy.deepcopy(netD)
		feature_adv = attack_method(netD_cp, feature, label, 'sign', coef_FGSM, num_iter, max_mag)
		mag = mag*i/(i+1) + torch.max(torch.abs(feature_adv-feature.cuda()), 1)[0].mean()/(i+1)
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
				optimizerD.param_groups[0]['lr'] /= 1.0
				#coef_FGSM *= 1.5
	if epoch % 5 == 0:
		print("===="*5)
		print("perturb mag: %.5f" %(mag))
		#dataset_adv = torch.load('./adv_exam/adv_gradient_FGSM_step1.pt')
		netD.eval()
		train_acc, train_loss = TestAcc_dataloader(netD, train_data,loss_func)
		valid_acc, valid_loss = TestAcc_dataloader(netD, valid_data,loss_func)
		test_acc, test_loss = TestAcc_dataloader(netD, test_data,loss_func)
		#test_adv_acc = TestAcc_tensor(netD, dataset_adv)
		print('[%d/%d]Clean accu: %.3f \t %.3f \t %.3f' %(epoch, epoch_num, train_acc, valid_acc, test_acc) )
		

		acc_tr, loss_tr = TestAdvAcc_dataloader(netD, train_data, 'sign', coef_FGSM, attack_method,loss_func, num_iter, max_mag)
		acc_valid, loss_valid = TestAdvAcc_dataloader(netD, valid_data, 'sign', coef_FGSM, attack_method,loss_func, num_iter, max_mag)
		acc_t, loss_t = TestAdvAcc_dataloader(netD, test_data, 'sign', coef_FGSM, attack_method,loss_func, num_iter, max_mag)
		print('[%d/%d]White-box Attack accuracy: %.3f \t %.3f \t %.3f' %(epoch, epoch_num,acc_tr, acc_valid, acc_t))
		

		#acquireAdvGradient(netD, train_data, test_data, loss_func)
		#acquireInputGradient(netD, valid_data, test_data, loss_func)
		print("Train Loss: %3.5f;Valid Loss: %3.5f; Test Loss: %3.5f" %(train_loss, valid_loss, test_loss))
		print("Train Loss: %3.5f;Valid Loss: %3.5f; Test Loss: %3.5f" %(loss_tr, loss_valid, loss_t))
		print("===="*5)
		
torch.save(netD.state_dict(), './netD_adv_tr.pkl')



