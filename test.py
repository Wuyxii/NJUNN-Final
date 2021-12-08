import torch
from matplotlib import pyplot as plt
from torch import nn
import sys


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


argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def plot(width, length, net_G):
    Z = torch.normal(0, 1, size=(width * length, 200, 1, 1), device=try_gpu())
    fake_x = net_G(Z).permute(0, 2, 3, 1) / 2 + 0.5
    imgs = torch.cat(
        [torch.cat([
            fake_x[i * length + j].cpu().detach() for j in range(length)], dim=1)
            for i in range(len(fake_x) // length)], dim=0)
    plt.imshow(imgs)
    plt.show()


if __name__ == "__main__":
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

    net_G.load_state_dict(torch.load(sys.argv[1]))
    net_G.to(try_gpu())
    plot(4, 4, net_G)