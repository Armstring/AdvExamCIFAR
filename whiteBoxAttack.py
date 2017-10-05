from __future__ import print_function
import argparse
import torch
#import torch.nn as nn
from nets.classifiers import _netD_mnist,_netG_mnist
from dataProcess.read_data import read_MNIST
from utils.utils import TestAcc_dataloader, TestAdvAcc_dataloader



batch_size = 64
test_batch_size = 1000
train_data , test_data = read_MNIST(batch_size, test_batch_size)

netD = _netD_mnist()
netD.cuda()
netD.load_state_dict(torch.load('netD_FGSM_step1.pkl'))
print('Test accuracy of netD: %.3f'%(TestAcc_dataloader(netD,test_data)))

print('White-box Attack training accuracy: %.3f' %(TestAdvAcc_dataloader(netD, train_data, 2, 2.5)))
print('White-box Attack test accuracy: %.3f' %(TestAdvAcc_dataloader(netD, test_data, 2, 2.5)))
	

