import torch
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist
import torch.nn as nn
import numpy as np
from random import Random
import torch.optim as optim
from math import ceil
import torch.nn.functional as F

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

    def __init__(self, data, sizes, seed=1234):
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

def partition_dataset(dataset):
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


def train(args, model, device, dataset, dataloader_kwargs):
    
    torch.manual_seed(1234)
    train_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    error = nn.NLLLoss()

    optimizer = optim.SGD(model.parameters(), momentum=args.momentum, lr=args.learning_rate)
    for epoch in range(args.num_epochs):
        epoch_loss = 0
        for idx, (images, labels) in enumerate(train_loader):

            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            output = model(images)

            loss = error(output, labels)
            epoch_loss += loss

            loss.backward()

            if idx % args.step_size == 0:
                for param in model.parameters():
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= args.world_size

            optimizer.step()

            if idx % args.display_interval == 0:
                print('Rank: {}, Trian Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:0.6f}'.format(
                    args.rank, epoch, idx * len(images), len(train_loader.dataset),
                    100 * idx / len(train_loader), loss.item()))
        return 

def test(args, model, device, dataset, dataloader_kwargs):
    
    test_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    error = nn.NLLLoss(reduction='sum')

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)

            loss = error(output, labels)
            test_loss = loss.item()
            pred = output.max(1)[1]
            correct += pred.eq(labels.to(device)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average Loss {:.2f}, Accuracy: {:.2f}%'.format(
        test_loss, correct / len(test_loader.dataset) * 100))


