from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from constants import *
from dataProcess.read_data import read_CIFAR10
from utils.utils import weights_init,weights_xavier_init

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.conv1a = nn.Conv2d(num_channel, 32, 3, 1, 1)
        self.conv1b = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2) #16*16
        self.conv2a = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv2b = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2) #8*8
        self.conv3a = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv3b = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2) #4*4
        self.fc1a = nn.Linear(128*16, h_dim)
        #self.fc1b = nn.Linear(1024, h_dim)
        self.fc2a = nn.Linear(128*16, h_dim)
        #self.fc2b = nn.Linear(1024, h_dim)


        self.fc3 = nn.Linear(h_dim, 4096) #64*8*8
        self.bn4 = nn.BatchNorm2d(64)
        self.upsample1 = nn.ConvTranspose2d(64, 64, 4, 2, 1) #32* 16*16
        self.conv4a = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv4b = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(32)
        self.upsample2 = nn.ConvTranspose2d(32, 16, 4, 2, 1) #16* 32*32
        self.conv5a = nn.Conv2d(16, 3, 3, 1, 1)
        self.conv5b = nn.Conv2d(3, 3, 3, 1, 1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.apply(weights_xavier_init)
    def encode(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1b(self.conv1a(x)))))
        x = self.pool2(self.relu(self.bn2(self.conv2b(self.conv2a(x)))))
        x = self.pool3(self.relu(self.bn3(self.conv3b(self.conv3a(x)))))
        x = x.view(x.size()[0],-1)
        h1 = self.fc1a(x)
        h2 = self.fc2a(x)
        return h1,h2

    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu

    def decode(self, z):
        z = self.fc3(z)
        z = z.view(z.size()[0], -1, 8, 8)
        z = self.relu(self.bn4(z))
        z = self.upsample1(z)
        z = self.upsample2(self.relu(self.bn5(self.conv4b(self.conv4a(z)))))
        z = self.conv5b(self.conv5a(z))
        return self.tanh(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    sampleloss = nn.BCELoss()
    pixelloss = nn.MSELoss()
    BCE = sampleloss((1.0+recon_x.view(-1,num_channel*image_size))/2.0, (1.0+x.view(-1, num_channel*image_size))/2.0)
    MSE = pixelloss(recon_x.view(-1,num_channel*image_size), x.view(-1, num_channel*image_size))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= batch_size * num_channel*image_size

    return BCE + KLD + MSE


def train(epoch):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = vae_loss(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx %100  == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        test_loss += vae_loss(recon_batch, data, mu, logvar).data[0]
        if i == 0 and epoch%10==1:
          n = min(data.size(0), 8)
          comparison = torch.cat([data[:n],
                                  recon_batch[:n]])
          save_image(comparison.data.cpu(),
                     'Vae_results/reconstruction_' + str(epoch) + '.png', normalize = True, nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    batch_size = 128
    test_batch_size = 128
    train_loader , valid_loader, test_loader = read_CIFAR10(batch_size, test_batch_size, 0.2)

    model = VAE()
    model.cuda()
    for epoch in range(1, epoch_num + 1):
        train(epoch)
        test(epoch)
        if epoch%10==1:
            sample = Variable(torch.randn(64, h_dim))
            sample = sample.cuda()
            sample = model.decode(sample).cpu()
            save_image(sample.data,
                       'Vae_results/sample_' + str(epoch) + '.png',normalize = True)







