# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
from constants import *
import torch.nn.init as init
import copy

def weights_init(layer):
  classname = layer.__class__.__name__
  if classname.find('Conv') != -1:
    layer.weight.data.normal_(0.0, 0.02)
  elif classname.find('Lin') != -1:
    layer.weight.data.normal_(0.0, 0.02)

def weights_xavier_init(layer):
  classname = layer.__class__.__name__
  if classname.find('Conv') != -1:
    layer.weight.data.normal_(0.0, 0.02)
  elif classname.find('Lin') != -1:
    init.xavier_uniform(layer.weight.data)


def accu(output, label):
  _, pred = torch.max(output, 1)
  return 1.0*(pred.data==label.data).sum()

def TestAdvAcc_dataloader(net, dataset, flag, p_coef, method, loss_func, num_iter, mag):
  adv_acc = .0
  num = 0
  net.eval()
  res = 0.0
  qq = 0
  for i,data_batch in enumerate(dataset):
    feature, label = data_batch
    netD_cp = copy.deepcopy(net)
    feature_adv = method(netD_cp, feature, label, flag, p_coef, num_iter, mag)
    feature_adv, label = Variable(feature_adv), Variable(label.cuda())
  
    outputs = net(feature_adv.detach())
    error = loss_func(outputs, label)
    res += error.data[0]
    adv_acc += accu(outputs, label)
    num += label.size()[0]
    qq+=1
  #print("%5.5f" %(res/qq))
  return (1.0*adv_acc/num, res/qq)

def TestAcc_dataloader(net, dataset, loss_func):
  acc = .0
  num = 0
  net.eval()
  res = 0.0
  qq = 0
  for i,data_batch in enumerate(dataset):
    feature, label = data_batch
    feature, label = Variable(feature.cuda()), Variable(label.cuda())
    outputs = net(feature)
    error = loss_func(outputs, label)
    acc += accu(outputs, label)
    res += error.data[0]
    num += label.size()[0]
    qq+=1
  return (1.0*acc/num, res/qq)

def TestAcc_tensor(net, dataset):
  dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(dataset[0], dataset[1]),
    batch_size=64, shuffle=True, drop_last = False)
  return TestAcc_dataloader(net, dataloader)

def TestAcc_tensorpath(netD, advpath):
  dataset = torch.load(advpath)
  return TestAcc_tensor(netD, dataset_adv)

def AdvAcc_Gnet(netD, netG, testset, coef):
  adv_acc = .0
  loss_func = nn.CrossEntropyLoss()
  for i,data_batch in enumerate(testset):
    #G_input.normal_(0,1)
    feature, label = data_batch
    feature, label = Variable(feature.cuda(),requires_grad = True), Variable(label.cuda())
    
    outputs = netD(feature)
    error = loss_func(outputs, label)
    error.backward()
    perturb = feature.grad
    perturb = torch.sign(perturb)
    
    adv_perb = netG(feature)
    fake = feature + coef*adv_perb
    outputs = netD(fake)
    adv_acc += accu(outputs, label)
  return adv_acc/(i+1)












