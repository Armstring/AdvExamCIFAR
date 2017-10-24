# -*- coding: utf-8 -*-
import torch
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np



def read_MNIST(batch_size, test_batch_size, valid_ratio, shuffle=True):
	train_dataset = datasets.MNIST('../data', train=True, transform=transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5,), (0.5,))
			]))
	valid_dataset = datasets.MNIST('../data', train=True, transform=transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5,), (0.5,))
			]))
	num_train = len(train_dataset)
	indices = list(range(num_train))
	split = int(np.floor(valid_ratio * num_train))
	if shuffle == True:
		#np.random.seed(random_seed)
		np.random.shuffle(indices)
	train_idx, valid_idx = indices[split:], indices[:split]
	train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
	valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)


	train_data = torch.utils.data.DataLoader(train_dataset,
		batch_size=batch_size, sampler=train_sampler, drop_last = True)
	valid_data = torch.utils.data.DataLoader(valid_dataset,
		batch_size=batch_size, sampler=valid_sampler, drop_last = True)

	test_data = torch.utils.data.DataLoader(
		datasets.MNIST('../data', train=False, transform=transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5,), (0.5,))
			])),
		batch_size=test_batch_size, shuffle=True, drop_last = True)
	return (train_data, valid_data, test_data)


def read_CIFAR10(batch_size, test_batch_size, valid_ratio, shuffle=True):
	train_dataset = datasets.CIFAR10('../data', train=True, transform=transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
			]),download=True)
	valid_dataset = datasets.CIFAR10('../data', train=True, transform=transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
			]),download=True)
	num_train = len(train_dataset)
	indices = list(range(num_train))
	split = int(np.floor(valid_ratio * num_train))
	if shuffle == True:
		#np.random.seed(random_seed)
		np.random.shuffle(indices)
	train_idx, valid_idx = indices[split:], indices[:split]
	train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
	valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)

	train_data = torch.utils.data.DataLoader(train_dataset,
		batch_size=batch_size, sampler=train_sampler, drop_last = True)
	valid_data = torch.utils.data.DataLoader(valid_dataset,
		batch_size=batch_size, sampler=valid_sampler, drop_last = True)
	

	test_data = torch.utils.data.DataLoader(
		datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
			]),download=True),
		batch_size=test_batch_size, shuffle=True, drop_last = True)
	return (train_data, valid_data, test_data)

