from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import torchvision.models

from nets.classifiers import _netD_cifar10
from constants import *
from utils.utils import accu,TestAcc_dataloader, TestAcc_tensor
from models.model_train import advexam_gap, advexam_ll, advexam_gradient
from dataProcess.read_data import read_CIFAR10


#torch.manual_seed(31415926)
#torch.cuda.manual_seed(31415926)
batch_size = 128
test_batch_size = 128
train_data , valid_data, test_data = read_CIFAR10(batch_size, test_batch_size, 0.2, False)
epoch_num = 30

netD = _netD_cifar10()
netD.cuda()
loss_func = nn.CrossEntropyLoss()

'''
##### Train netD on real data
#optimizerD = optim.Adam(netD.parameters(), lr=lr_D, betas=(0.9, 0.999))
optimizerD = optim.SGD(netD.parameters(), lr=0.02, weight_decay = 0.01)

for epoch in range(epoch_num):
  running_loss = .0
  running_acc = .0
  netD.train()
  for i,data_batch in enumerate(train_data):
    netD.zero_grad()
    feature,label = data_batch
    feature_v, label_v = Variable(feature.cuda()), Variable(label.cuda())
    outputs_v = netD(feature_v)
    loss = loss_func(outputs_v, label_v)
    _, pred = torch.max(outputs_v, 1)
    loss.backward()
    optimizerD.step()
    running_loss += loss.data[0]
    running_acc += accu(outputs_v, label_v)/batch_size
    if i % 100 ==99:
      print('[%d, %5d] loss: %.3f accuracy: %.3f' %(epoch + 1, i + 1,
      	running_loss / 100, running_acc/100))
      running_loss = .0
      running_acc = .0
  print('Validation accuracy of netD: %.3f'%(TestAcc_dataloader(netD,valid_data, loss_func)[0]))
  if epoch in {10,20}:
    optimizerD.param_groups[0]['lr'] /= 4.0
  

print('Finished Pre_train netD')
torch.save(netD.state_dict(), './netD.pkl')
print('Test accuracy of netD: %.3f'%(TestAcc_dataloader(netD,test_data, loss_func)[0]))

'''
netD.load_state_dict(torch.load('netD.pkl'))
print('Test accuracy of netD: %.3f'%(TestAcc_dataloader(netD,test_data, loss_func)[0]))

######################
## gap base

flag = 'sign'
step_num = 10
path = "./adv_exam/"
coef = 0.005
max_per = 0.03
print('gradient', flag, coef, step_num)

loss_res = []
acc_res = []
perb_res = []
num_trail=20

for j in range(num_trail):
  mag = .0
  adv_list = []
  label_list = []
  for i,data_batch in enumerate(train_data):
    feature, label = data_batch
    feature_temp = feature[:].cuda()
    perb_temp = advexam_gradient(netD, feature, label, flag, coef, step_num, max_per)
    if flag == 2:
      mag += torch.norm((feature_temp-perb_temp).view(64,-1), 2, 1).mean()
    else:
      mag += torch.max(torch.abs(feature_temp-perb_temp), 1)[0].mean()

    adv_list.append(perb_temp)
    label_list.append(label)

  print(1.0*mag/i)
  adv_featureset = torch.cat(adv_list, 0)
  labelset = torch.cat(label_list, 0)
  res = TestAcc_tensor(netD,(adv_featureset, labelset), loss_func)
  print('Adv accuracy of netD: %.3f;\t %.5f'%(res[0], res[1]))
  loss_res.append(res[1])
  acc_res.append(res[0])


print("====="*5)
print(torch.mean(loss_res), torch.var(loss_res))
print(torch.mean(acc_res), torch.var(acc_res))


#if flag=='sign':
#  torch.save((adv_featureset, labelset), path+'adv_gradient_FGSM_step%d.pt'%(step_num))
#else:
#  torch.save((adv_featureset, labelset), path+'adv_gradient_L%d_step%d.pt'%(flag, step_num))

