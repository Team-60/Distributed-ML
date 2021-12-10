import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import os

def train(args, model, device, dataset, dataloader_kwargs):
    
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
            SGD_step(optimizer)

            if idx % args.display_interval == 0:
                print('Process id: {}, Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:0.6f}'.format(
                    os.getpid(), epoch, idx * len(images), len(train_loader.dataset),
                    100 * idx / len(train_loader), loss.item()))


def test(args, model, device, dataset, dataloader_kwargs):
    
    test_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    error = nn.NLLLoss(reduction='sum')

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for (images, labels) in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)

            loss = error(output, labels)
            test_loss += loss.item()
            pred = output.max(1)[1]
            correct += pred.eq(labels.to(device)).sum().item()

    test_loss /= (len(test_loader.dataset) * args.batch_size)
    print('Test set: Average Loss {:.6f}, Accuracy: {:.2f}%'.format(
        test_loss, correct / len(test_loader.dataset) * 100))


def SGD_step(optimizer):

    if not isinstance(optimizer, optim.SGD):
        raise ValueError("Non SGD optimizer used in SGD_step()")

    loss = None

    for group in optimizer.param_groups:
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']
        lr = -group['lr']

        for p in group['params']:
            if p.grad is not None:
                p_grad = p.grad.data
                if momentum != 0:
                    param_state = optimizer.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = p_grad.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, p_grad)
                    if nesterov:
                        p_grad = p_grad.add(momentum, buf)
                    else:
                        p_grad = buf

                p.data.add_(lr, p_grad)

    return loss

