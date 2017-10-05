# -*- coding: utf-8 -*-
import torch
import torch.utils.data
from torchvision import datasets, transforms




def read_MNIST(batch_size, test_batch_size):
	train_data = torch.utils.data.DataLoader(
		datasets.MNIST('../data', train=True, transform=transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5,), (0.5,))
			])),
		batch_size=batch_size, shuffle=True, drop_last = True)
	test_data = torch.utils.data.DataLoader(
		datasets.MNIST('../data', train=False, transform=transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5,), (0.5,))
			])),
		batch_size=test_batch_size, shuffle=True, drop_last = True)
	return (train_data, test_data)


def read_CIFAR10(batch_size, test_batch_size):
	train_data = torch.utils.data.DataLoader(
		datasets.CIFAR10('../data', train=True, transform=transforms.Compose([
			transforms.ToTensor(),
    		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
			]),download=True),
		batch_size=batch_size, shuffle=True, drop_last = True)
	test_data = torch.utils.data.DataLoader(
		datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
			transforms.ToTensor(),
     		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
			]),download=True),
		batch_size=test_batch_size, shuffle=True, drop_last = True)
	return (train_data, test_data)