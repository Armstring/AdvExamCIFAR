from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.optim as optim

from nets.classifiers import _netD_cifar10,_netG_cifar10
from constants import *
import models.model_train as model_train
from dataProcess.read_data import read_CIFAR10
import glob

#torch.manual_seed(31415926)
#torch.cuda.manual_seed(31415926)

train_data , test_data = read_CIFAR10(batch_size, test_batch_size)


#netD = _netD_mnist()
#netD.cuda()
#netD.load_state_dict(torch.load('netD.pkl'))
#print('Test accuracy of netD: %.3f'%(TestAcc(netD,test_data)))
########################

model_train.adv_train_gradient(train_data, test_data, 'sign', coef_FGSM, 1)
#model_train.adv_train_gradient(train_data, test_data, 'sign', coef_FGSM, 3)
#model_train.adv_train_gradient(train_data, test_data, 2, coef_L2, 1)
#model_train.adv_train_gradient(train_data, test_data, 2, coef_L2, 3)
#model_train.adv_train_GAN(train_data, test_data)
