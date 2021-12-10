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
from train import train, test

import warnings
warnings.filterwarnings("ignore") 


parser = argparse.ArgumentParser(description='Shared Memory MNIST')
parser.add_argument('--batch-size', type=int, default=64, help='batch size for training (default: 64)')
parser.add_argument('--num-epochs', type=int, default=100, help='number of epochs for training (default: 100)')
parser.add_argument('--momentum', type=int, default=0.9, help='momentum value of optimizer training (default: 0.9)')
parser.add_argument('--learning-rate', type=int, default=0.001, help='learning rate of optimizer training (default: 0.001)')
parser.add_argument('--display-interval', type=int, default=5, help='interval for printing while training each epoch')
parser.add_argument('--world-size', type=int, default=2, help='the world size of the cluster')
parser.add_argument('--rank', type=int, default=0, help='rank of the process')
parser.add_argument('--step-size', type=int, default=5, help='step size for local sgd')
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

def subsample(dataset, c):
  '''
  subsamples dataset by a factor of c
  '''
  sample_cnt = int(len(dataset) / (10 * c))
  class_seen = dict([(i, 0) for i in range(10)])
  train_indices = []
  for i, (_, l) in enumerate(dataset):
    if class_seen[l] >= sample_cnt:
      continue
    class_seen[l] += 1
    train_indices.append(i)
  return train_indices

def run(args, rank, size):

    train_set = torchvision.datasets.CIFAR10('./data', download=True, train=True, transform=transforms.Compose([transforms.ToTensor()]))
    test_set = torchvision.datasets.CIFAR10("./data", download=True, train=False, transform=transforms.Compose([transforms.ToTensor()])) 

    # sampling for smaller dataset: train
    train_set = torch.utils.data.Subset(train_set, subsample(train_set, 2))
    # sampling for smaller dataset: test
    test_set = torch.utils.data.Subset(test_set, subsample(test_set, 3))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = CNN()
    model.to(device)

    dataloader_kwargs = {
            'batch_size': args.batch_size,
            'shuffle': True,
    }
    train(args, model, device, train_set, dataloader_kwargs)

    if args.rank == 0:
        test(args, model, device, test_set, dataloader_kwargs)

    
def init_process(args, rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(args, rank, size)


if __name__ == "__main__":
    processes = []
    mp.set_start_method("spawn")
    init_process(args, args.rank, args.world_size, run)
