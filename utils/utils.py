# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
from constants import *
import torch.nn.init as init

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

def TestAdvAcc_dataloader(net, dataset, flag, p_coef):
  adv_acc = .0
  num = 0
  net.eval()
  for i,data_batch in enumerate(dataset):
    feature, label = data_batch

    perturb = torch.zeros(feature.size()).cuda()
    feature, label = Variable(feature.cuda()), Variable(label.cuda())
    perturb = Variable(perturb, requires_grad = True)
    feature_adv = feature + perturb

    outputs = net(feature_adv)
    error = .0
    for j in range(outputs.size()[0]):
      error -= outputs[j][label.data[j]]
    error.backward()
    _, pred = torch.max(outputs, 1)
    
    for j,image in enumerate(feature):
      if pred[j]==label[j]:
        if flag=='sign':
          perturb[j] +=  p_coef*torch.sign(perturb.grad[j])
        else:
          duel_norm = flag/(flag-1.0)
          perturb[j] += p_coef * perturb.grad[j]/torch.norm(perturb.grad[j].data, p = duel_norm)
    feature_adv = feature + perturb
    feature_adv.data.clamp_(min=-1.0, max = 1.0)
    outputs = net(feature_adv.detach())

    adv_acc += accu(outputs, label)
    num += label.size()[0]
  return 1.0*adv_acc/num

def TestAcc_dataloader(net, dataset):
  acc = .0
  num = 0
  net.eval()
  for i,data_batch in enumerate(dataset):
    feature, label = data_batch
    feature, label = Variable(feature.cuda()), Variable(label.cuda())
    outputs = net(feature)
    acc += accu(outputs, label)
    num += label.size()[0]
  return 1.0*acc/num

def TestAcc_tensor(net, dataset):
  dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(dataset[0], dataset[1]),
    batch_size=64, shuffle=True, drop_last = False)
  return TestAcc_dataloader(net, dataloader)

def TestAcc_tensorpath(netD, advpath):
  dataset = torch.load(advpath)
  return TestAcc_tensor(netD, dataset_adv)

def AdcAcc_Gnet(netD, netG, testset, coef):
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












