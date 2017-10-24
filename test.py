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

#import matplotlib.pyplot as plt

from nets.classifiers import _netD_cifar10,_netG_cifar10
from constants import *
from utils.utils import accu,TestAcc_dataloader, TestAcc_tensor, TestAdvAcc_dataloader
import models.model_train as model_train
from dataProcess.read_data import read_CIFAR10
import glob
import copy


def AdvAcc_dataloader(net, feature, label, flag, p_coef, method, loss_func, step):
  net.eval()
  
  random_per = torch.zeros(feature.size()).cuda()
  random_per.uniform_(0, 1)
  random_per.bernoulli_()
  random_per = (random_per-0.5)*2
  #print(list(random_per[14][2][26]))
  feature_noise = feature.cuda() + 0.03*random_per

  mag = torch.norm(random_per.view(64, -1))
  netD_cp = copy.deepcopy(net)
  feature_adv = method(netD_cp, feature_noise, label, flag, p_coef, step)
  feature_adv, label = Variable(feature_adv), Variable(label.cuda())
  
  outputs = net(feature_adv.detach())
  error = loss_func(outputs, label)
  res = error.data[0]
  adv_acc = accu(outputs, label)/label.size()[0]

  return (adv_acc, res, mag)

batch_size = 64
train_data , test_data = read_CIFAR10(batch_size, batch_size)

num_trials = 50
flag = 'sign'
path = "./adv_exam/"
loss_func = nn.CrossEntropyLoss()
method = model_train.advexam_gradient


netD = _netD_cifar10()
netD.cuda()

netD.load_state_dict(torch.load('netD.pkl'))
print('Test accuracy of netD: %.3f'%(TestAcc_dataloader(netD,test_data)))

res_acc_tr = []
res_loss_tr = []
res_acc_t = []
res_loss_t = []
feature_tr, label_tr, feature_t, label_t = None, None, None, None
for i,data_batch in enumerate(train_data):
    feature_tr, label_tr = data_batch 
    break
for i,data_batch in enumerate(test_data):
    feature_t, label_t = data_batch 
    break

for j in range(num_trials):
    res_tr = AdvAcc_dataloader(netD, feature_tr, label_tr, flag, 0.01, method, loss_func, 15)
    res_t = AdvAcc_dataloader(netD, feature_t, label_t, flag, 0.01, method, loss_func, 15)
    res_acc_tr.append(res_tr[0])
    res_loss_tr.append(res_tr[1])
    res_acc_t.append(res_t[0])
    res_loss_t.append(res_t[1])
    #print("[%3d] Training mag: %2.3f; Test mag: %2.3f" %(j,res_tr[2], res_t[2]))
    print("[%3d] Training: %.4f \t %.4f; Test: %.4f \t %.4f;" %(j, res_tr[0],res_tr[1], res_t[0],res_t[1]))





