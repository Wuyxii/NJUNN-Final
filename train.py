import torch
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils import data
from torch import nn
import time
from IPython import display
import seaborn as sns
import tensorflow as tf
import numpy as np
import random
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch import autograd
from torchviz import make_dot

# net constructing
class G_block(nn.Module):
    def __init__(self, out_channels, in_channels=3, kernel_size=4, strides=2,
                 padding=1, **kwargs):
        super(G_block, self).__init__(**kwargs)
        self.conv2d_trans = nn.ConvTranspose2d(in_channels, out_channels,
                                               kernel_size, strides, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d_trans(X)))


class D_block(nn.Module):
    def __init__(self, out_channels, in_channels=3, kernel_size=4, strides=2,
                padding=1, alpha=0.2, **kwargs):
        super(D_block, self).__init__(**kwargs)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
                                strides, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(alpha, inplace=True)

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d(X)))


# train
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def __getitem__(self, idx):
        return self.data[idx]


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def compute_gradient_penalty(real_images, fake_images, net_D):
    eta = torch.FloatTensor(batch_size, 1, 1, 1).uniform_(0, 1).to(try_gpu())
    eta = eta.expand(batch_size, real_images.size(1), real_images.size(2), real_images.size(3))

    interpolated = eta * real_images + ((1 - eta) * fake_images).to(try_gpu())
    interpolated = Variable(interpolated, requires_grad=True)

    prob_interpolated = net_D(interpolated)

    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                              grad_outputs=torch.ones(prob_interpolated.size()).to(try_gpu()),
                              create_graph=True, retain_graph=True)[0]

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_term
    return grad_penalty


def update_D(X, Z, net_D, net_G, loss, trainer_D, gan_type):
    batch_size = X.shape[0]
    ones = torch.ones((batch_size,), device=X.device)
    zeros = torch.zeros((batch_size,), device=X.device)
    trainer_D.zero_grad()
    real_Y = net_D(X)
    fake_X = net_G(Z)
    fake_Y = net_D(fake_X.detach())
    loss_D = (loss(real_Y, ones.reshape(real_Y.shape)) +
              loss(fake_Y, zeros.reshape(fake_Y.shape))) / 2

    if gan_type == 1:
        loss_D = loss_D + compute_gradient_penalty(X, fake_X.detach(), net_D)
    loss_D.backward()
    trainer_D.step()

    return loss_D


def update_G(Z, net_D, net_G, loss, trainer_G, gan_type):
    batch_size = Z.shape[0]
    ones = torch.ones((batch_size,), device=Z.device)
    trainer_G.zero_grad()
    fake_X = net_G(Z)
    fake_Y = net_D(fake_X)
    loss_G = loss(fake_Y, ones.reshape(fake_Y.shape))
    loss_G.backward()
    trainer_G.step()

    return loss_G


# plotting
def plot1(width, length, epoch, net_G):
    Z = torch.normal(0, 1, size=(width * length, latent_dim, 1, 1), device=try_gpu())
    fake_x = net_G(Z).permute(0, 2, 3, 1) / 2 + 0.5
    imgs = torch.cat(
        [torch.cat([
            fake_x[i * length + j].cpu().detach() for j in range(length)], dim=1)
            for i in range(len(fake_x) // length)], dim=0)
    plt.imshow(imgs)
    plt.savefig("epoch" + str(epoch) + ".jpg")
    plt.show()


def plot2(num, net_G):
    for i in range(0, num):
        Z = torch.normal(0, 1, size=(1, latent_dim, 1, 1), device=try_gpu())
        fake_x = net_G(Z).permute(0, 2, 3, 1) / 2 + 0.5
        imgs = torch.cat(
            [torch.cat([
                fake_x[i + j].cpu().detach() for j in range(1)], dim=1)
                for i in range(len(fake_x))], dim=0)
        plt.imshow(imgs)
        plt.show()


def train(net_D, net_G, data_iter, num_epochs, lr, latent_dim, gan_type,
          device=try_gpu()):
    print('training on', device)
    loss = nn.BCEWithLogitsLoss(reduction='sum')
    for w in net_D.parameters():
        nn.init.normal_(w, 0, 0.02)
    for w in net_G.parameters():
        nn.init.normal_(w, 0, 0.02)
    net_D, net_G = net_D.to(device), net_G.to(device)
    if gan_type == 1:
        trainer_hp = {'lr': lr, 'betas': [0.5, 0.999]}
        trainer_D = torch.optim.Adam(net_D.parameters(), **trainer_hp)
        trainer_G = torch.optim.Adam(net_G.parameters(), **trainer_hp)
    else:
        trainer_D = torch.optim.SGD(net_D.parameters(), lr=lr)
        trainer_G = torch.optim.SGD(net_G.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        metric = Accumulator(3)
        for X, _ in data_iter:
            if X.shape[0] != batch_size:
                continue
            Z = torch.normal(0, 1, size=(batch_size, latent_dim, 1, 1))
            X, Z = X.to(device), Z.to(device)
            metric.add(update_D(X, Z, net_D, net_G, loss, trainer_D, gan_type),
                       update_G(Z, net_D, net_G, loss, trainer_G, gan_type),
                       batch_size)

        loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]
        D_loss.append(loss_D)
        G_loss.append(loss_G)
        plot1(4, 4, epoch, net_G)
        print("Epoch: {}/{}, Discriminator loss: {}, Generator loss: {}".format(epoch, num_epochs, loss_D, loss_G))
    torch.save(net_G.state_dict(), "Net_G.param")
    torch.save(net_D.state_dict(), "Net_D.param")
    plt.plot(D_loss, label="Discriminator")
    plt.plot(G_loss, label="Generator")
    plt.xlabel("epoch")
    plt.legend(loc='best')
    plt.savefig("Result.jpg")
    plt.show()


if __name__ == "__main__":
    # data loading and preprocessing
    batch_size = 512
    train_file = sys.argv[1]
    pics = torchvision.datasets.ImageFolder(train_file)
    transformer = torchvision.transforms.Compose([
        torchvision.transforms.Resize((64, 64)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(0.5, 0.5)
    ])
    pics.transform = transformer
    data_iter = torch.utils.data.DataLoader(pics, batch_size=batch_size, shuffle=False)

    n_G = 64
    net_G = nn.Sequential(
        G_block(in_channels=200, out_channels=n_G * 8,
                strides=1, padding=0),
        G_block(in_channels=n_G * 8, out_channels=n_G * 4),
        G_block(in_channels=n_G * 4, out_channels=n_G * 2),
        G_block(in_channels=n_G * 2, out_channels=n_G),
        nn.ConvTranspose2d(in_channels=n_G, out_channels=3,
                           kernel_size=4, stride=2, padding=1, bias=False),
        nn.Tanh())

    n_D = 64
    net_D = nn.Sequential(
        D_block(n_D),
        D_block(in_channels=n_D, out_channels=n_D * 2),
        D_block(in_channels=n_D * 2, out_channels=n_D * 4),
        D_block(in_channels=n_D * 4, out_channels=n_D * 8),
        nn.Conv2d(in_channels=n_D * 8, out_channels=1,
                  kernel_size=4, bias=False))

    D_loss, G_loss = [], []
    latent_dim, num_epochs, lambda_term = 200, 30, 10
    if sys.argv[2] == 0:
        lr = 0.005
    else:
        lr = 0.001
    train(net_D, net_G, data_iter, num_epochs, lr, latent_dim, sys.argv[2])
