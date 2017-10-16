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
import vae
#torch.manual_seed(31415926)
#torch.cuda.manual_seed(31415926)
batch_size = 64
test_batch_size = 128
train_data , test_data = read_CIFAR10(batch_size, test_batch_size)


netD = _netD_cifar10()
netD.cuda()
netD.load_state_dict(torch.load('netD.pkl'))
#print('Test accuracy of netD: %.3f'%(TestAcc_dataloader(netD,test_data)))
loss_func = nn.CrossEntropyLoss()
#netD_cp = copy.deepcopy(netD)
netD.train()
print("===="*5)
#acquireGradient(netD, train_data, test_data, loss_func)
acquireInputGradient(netD, train_data, test_data, loss_func)
#print("===="*5)
#acquireAdvGradient(netD, train_data, test_data, loss_func)

netD = _netD_cifar10()
netD.cuda()
optimizerD = optim.SGD(netD.parameters(), lr=0.01, weight_decay = 0.0002)
