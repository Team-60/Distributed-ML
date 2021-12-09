import torch
import torchvision
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import torch.distributed as dist
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import argparse
import os
from random import Random
import torch.optim as optim
from math import ceil
import torch.nn.functional as F


parser = argparse.ArgumentParser(description='Shared Memory MNIST')
parser.add_argument('--batch-size', type=int, default=64, help='batch size for training (default: 64)')
parser.add_argument('--num-epochs', type=int, default=100, help='number of epochs for training (default: 100)')
parser.add_argument('--momentum', type=int, default=0.9, help='momentum value of optimizer training (default: 0.9)')
parser.add_argument('--learning-rate', type=int, default=0.001, help='learning rate of optimizer training (default: 0.001)')
parser.add_argument('--display-interval', type=int, default=5, help='interval for printing while training each epoch')
parser.add_argument('--world-size', type=int, default=2, help='the world size of the cluster')
parser.add_argument('--rank', type=int, default=0, help='rank of the process')
args = parser.parse_args()

class CNN(nn.Module):

  def __init__(self):
    super().__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
      nn.MaxPool2d(kernel_size=2),
      nn.ReLU(),
    )
    self.lin = nn.Sequential(
        nn.Linear(in_features=32*15*15, out_features=600),
        nn.ReLU(),
        nn.Linear(in_features=600, out_features=120),
        nn.ReLU(),
        nn.Linear(in_features=120, out_features=10)
    )
    self.softmax = nn.LogSoftmax(dim=1)

  def forward(self, x):
    out = self.conv(x)
    out = out.view(out.size(0), -1)
    out = self.lin(out)
    out = self.softmax(out)
    return out


""" Dataset partitioning helper """
class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

""" Partitioning MNIST """
def partition_dataset():
    train_set = torchvision.datasets.CIFAR10('./data', download=True, train=True, transform=transforms.Compose([transforms.ToTensor()]))
    size = dist.get_world_size()
    bsz = 128 / float(size)
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(train_set, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition,
                                         batch_size=int(bsz),
                                         shuffle=True)
    return train_set, bsz


def run(rank, size):
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    model = CNN()
    optimizer = optim.SGD(model.parameters(),
                          lr=0.01, momentum=0.5)

    num_batches = ceil(len(train_set.dataset) / float(bsz))
    for epoch in range(10):
        epoch_loss = 0.0
        for data, target in train_set:
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()
        print('Rank ', dist.get_rank(), ', epoch ',
              epoch, ': ', epoch_loss / num_batches)

""" Gradient averaging. """
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    processes = []
    mp.set_start_method("spawn")

    p = mp.Process(target=init_process, args=(args.rank, args.world_size, run))
    p.start()
    processes.append(p)
    p.join()
