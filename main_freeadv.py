'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import math
os.environ["CUDA_VISIBLE_DEVICES"]="6"
import argparse

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--k', default=10, type=int, help='learning rate')
parser.add_argument('--iter_eps', default=2/255, type=float, help='learning rate')
parser.add_argument('--max_eps', default=8/255, type=float, help='learning rate')
parser.add_argument('--mode', default='accumulate', help='learning rate')
parser.add_argument('--scale_lambda', '-s', action='store_true', help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--warm', '-w', action='store_true', help='resume from checkpoint')
parser.add_argument('--test', '-t', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

def random_noise(im, type='uniform',p=0.5, eps=4 / 255):
    """
    Adds a Gaussian noise to an input image with a chance of p
 
    Args:
        im: (Tensor) Image of size [B, C, H, W]
        p: (float) Probability of adding noise
        eps: (float) std of Gaussian noise added
    """
    random_p = torch.zeros(im.size(0)).bernoulli_(p=p).view(-1, 1, 1, 1)

    if type == 'gaussian':
        noise = torch.zeros_like(im).normal_(0, eps)
    elif type == 'uniform':
        noise = torch.zeros_like(im).uniform_(-eps, eps)
    
    if im.is_cuda:
        random_p = random_p.cuda()
        noise = noise.cuda()
    
    im = im + random_p * noise
    return im.clamp(0, 1)

MEAN = torch.Tensor([0.4914, 0.4822, 0.4465])
STD = torch.Tensor([0.2023, 0.1994, 0.2010])


def unnormalize(im):
    mean = MEAN.cuda() if im.is_cuda else MEAN
    std = STD.cuda() if im.is_cuda else STD

    if im.dim() == 4:
        im = im.transpose(1, 3)
        im = im * std + mean
        im = im.transpose(1, 3)
    else:
        im = im.transpose(0, 2)
        im = im * std + mean
        im = im.transpose(0, 2)

    return im


def normalize(im):
    mean = MEAN.cuda() if im.is_cuda else MEAN
    std = STD.cuda() if im.is_cuda else STD

    if im.dim() == 4:
        im = im.transpose(1, 3)
        im = (im - mean) / std
        im = im.transpose(1, 3)
    else:
        im = im.transpose(0, 2)
        im = (im - mean) / std
        im = im.transpose(0, 2)

    return im


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18_bn()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume or args.test:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt_%s_%d_scale%d_warm%d.t7'%(args.mode, args.k, args.scale_lambda, args.warm))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    eps = torch.zeros([128,3,32,32]).cuda()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if eps.size(0) != inputs.size(0):
            eps = eps[:inputs.size(0)]
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        if args.warm and epoch < warm_epoch: 
            num_iter = 1
        else:
            num_iter = args.k

        for j in range(num_iter):
            eps.requires_grad_()
            outputs = net(normalize((inputs+eps).clamp(0,1)))
            if args.mode == 'accumulate':
                loss = criterion(outputs, targets) 
                if args.scale_lambda:
                    loss /= num_iter
            else:
                loss = criterion(outputs, targets)

            loss.backward()
            eps = (eps + args.iter_eps* eps.grad.data.sign()).clamp(-args.max_eps, args.max_eps).detach()
            if args.mode != 'accumulate':
                optimizer.step()
                optimizer.zero_grad()

        if args.mode == 'accumulate':
            optimizer.step()
            optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(normalize(inputs))
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    if not args.test:
        # Save checkpoint.
        acc = 100.*correct/total
#         if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_%s_%d_scale%d_warm%d.t7'%(args.mode, args.k, args.scale_lambda, args.warm))
        best_acc = acc


if args.test:
    test(0)
else:
    max_epoch = int(200 / args.k)
#     warm_epoch = int(max_epoch / 10)
    warm_epoch = 10
    if args.warm:
        max_epoch += math.ceil(((args.k-1)/args.k)*warm_epoch) #add back number of iteration loss from using warm restart

    for epoch in range(start_epoch, max_epoch):
       if epoch == int(max_epoch/2):
           for g in optimizer.param_groups:
               g["lr"] = args.lr /10
       train(epoch)
       test(epoch)
