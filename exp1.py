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
from scipy.optimize import fmin_l_bfgs_b
#import matplotlib.pyplot as plt

'''
This is to test if the inner maximazition can be solved optimally, by 
checking the loss values of the adv examples generated from multiple 
random perturbation of the original images. 

We also test after the adv training of the network on the FIXED images,
if it is still possible to find adversarial exmaples for both the vanilla network
and the adv_train network.
'''
batch_size = 128
test_batch_size = 128
train_data , valid_data, test_data = read_CIFAR10(batch_size, test_batch_size, 0.2)
attack_method = model_train.advexam_gradient

netD = _netD_cifar10()
netD.cuda()
netD.load_state_dict(torch.load('netD.pkl'))
loss_func = nn.CrossEntropyLoss()
optimizerD = optim.SGD(netD.parameters(), lr=0.01, weight_decay = 0.0002)
epoch_num = 70

image_batch = None
label_batch = None
for i, data_batch in enumerate(train_data):
	image_batch,label_batch = data_batch
	image_batch, label_batch = Variable(image_batch.cuda()), Variable(label_batch.cuda())
	break

#print(image_batch.data.cpu().numpy().reshape(-1))

#print(adv_per0.shape)

def f(perturb):
	perturb = torch.from_numpy(perturb)
	perturb = perturb.view(batch_size, num_channel, image_shape[0], image_shape[1])
	perturb = perturb.type(torch.FloatTensor)
	perturb_var = Variable(perturb.cuda(), requires_grad = True)
	feature = image_batch + perturb_var
	netD.eval()
	outputs = netD(feature)
	error = -loss_func(outputs, label_batch)
	error.backward()
	return error.data.cpu().numpy()[0], perturb_var.grad.data.cpu().view(-1).numpy().astype('double')

num_trial = 30
res = np.zeros(num_trial)
for j in range(num_trial):
	print("Trial: %2d" %(j))
	adv_per0 = np.zeros(image_batch.view(-1).size()[0])
	bound = [(-coef_FGSM,coef_FGSM)] * adv_per0.shape[0]
	adv_per = fmin_l_bfgs_b(f,  adv_per0,  pgtol = 1e-12, bounds = bound, maxiter=15000, disp =None)
	res[j] = adv_per[1]
	print("[%2d/%2d] Convergence: %d; Iterations: %d" % (j, num_trial, adv_per[2]['warnflag'], adv_per[2]['nit']))
	print("max perturbation: %.3f; min perturbation: %.3f" %(np.max(adv_per[0]), np.min(adv_per[0])))
print(np.var(res), np.mean(res))



