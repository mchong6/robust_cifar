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
os.environ["CUDA_VISIBLE_DEVICES"]="4"
import argparse

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--k', default=20, type=int, help='learning rate')
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
net = ResNet18()
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
    checkpoint = torch.load('./checkpoint/ckpt_sub_%s_%d_scale%d_warm%d.t7'%(args.mode, args.k, args.scale_lambda, args.warm))
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
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        eps = torch.zeros_like(inputs).cuda()

        if args.warm and epoch < warm_epoch: 
            num_iter = 1
        else:
            num_iter = args.k

        for j in range(num_iter):
            optimizer.zero_grad()
#             inputs.requires_grad_()
#             outputs = net(normalize(inputs))
            eps.requires_grad_()
            outputs = net(normalize((inputs+eps).clamp(0,1)))
            loss = criterion(outputs, targets) 

            loss.backward()
#             mask = (inputs.grad.data.abs() > 1e-7).float()
#             inputs = (inputs + mask*torch.zeros_like(inputs).uniform_(-2*args.max_eps, 2*args.max_eps)).clamp(0,1).detach()
            
#             eps = (eps - args.iter_eps* eps.grad.data.sign()).clamp(-args.max_eps, args.max_eps).detach()

        optimizer.zero_grad()
#         mask = (eps.abs() > 1e-5).float()
        threshold = eps.grad.data.abs()
        mask = (threshold < torch.median(threshold)).float()
        mask_ratio = (mask.sum()/mask.numel())
#         mask_ratio = (mask.sum())
        inputs = mask*inputs
#         inputs = (inputs + mask*torch.zeros_like(inputs).uniform_(-2*args.max_eps, 2*args.max_eps)).clamp(0,1).detach()
        outputs = net(normalize(inputs))
        loss = criterion(outputs, targets) 

        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Mask: %.3f'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, mask_ratio))

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
        torch.save(state, './checkpoint/ckpt_sub_%s_%d_scale%d_warm%d.t7'%(args.mode, args.k, args.scale_lambda, args.warm))
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
