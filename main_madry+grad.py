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
os.environ["CUDA_VISIBLE_DEVICES"]="5"
import argparse

from models import *
from utils import progress_bar
from advertorch.attacks import LinfPGDAttack
from advertorch.utils import NormalizeByChannelMeanStd
from torch.utils.tensorboard import SummaryWriter


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

os.makedirs('logs', exist_ok=True)
writer = SummaryWriter("logs/test2")

# Model
MEAN = torch.Tensor([0.4914, 0.4822, 0.4465])
STD = torch.Tensor([0.2023, 0.1994, 0.2010])
norm = NormalizeByChannelMeanStd(mean=MEAN, std=STD)

print('==> Building model..')
net = torch.nn.DataParallel(nn.Sequential(norm, ResNet18()).cuda())
cudnn.benchmark = True

if args.resume or args.test:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#     checkpoint = torch.load('./checkpoint/ckpt_%s_%d_scale%d_warm%d.t7'%(args.mode, args.k, args.scale_lambda, args.warm))
    checkpoint = torch.load('./checkpoint/ckpt_madry+grad.t7')
    net.load_state_dict(checkpoint['net'])
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

adversary = LinfPGDAttack(
    net, eps=8/255, eps_iter=2/255, nb_iter=7,
    rand_init=True, targeted=False)

def cw(output, targets):
    y_onehot = torch.FloatTensor(targets.size(0), 10).cuda().zero_()
    y_onehot.scatter_(1, targets.unsqueeze(1), 1)
    real = (y_onehot * output).sum(dim=1)

    other = ((1.0 - y_onehot) * output - (y_onehot * 10000.0)
             ).max(1)[0]
    # - (y_onehot * TARGET_MULT) is for the true label not to be selected

    loss = (real-other + 20).clamp(min=0.)
    return loss.sum()

def calc_grad(inputs, targets, classifier):
    inputs.requires_grad_()
    with torch.enable_grad():
        output = classifier(inputs)
        loss = criterion(output, targets)
#         cw_loss = cw(output, targets)
        
    grad = torch.autograd.grad(loss, [inputs], create_graph=True)[0]
    grad_loss = grad.norm(2)
    return loss, 2*grad_loss, output

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    grad_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        adv_im = adversary.perturb(inputs, targets).detach()
        loss, grad_loss, outputs = calc_grad(adv_im, targets, net)
#         outputs = net(adv_im)
#         loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        writer.add_scalar("loss", loss.item(), len(trainloader)*epoch+batch_idx)
        writer.add_scalar("grad loss", grad_loss.item(), len(trainloader)*epoch+batch_idx)

        train_loss += loss.item()
        grad_loss += grad_loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Grad Loss: %.6f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), grad_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
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
        torch.save(state, './checkpoint/ckpt_madry+grad2.t7')
        best_acc = acc


if args.test:
    test(0)
else:
    max_epoch = 200
    for epoch in range(start_epoch, max_epoch):
       if epoch == int(max_epoch/2):
           for g in optimizer.param_groups:
               g["lr"] = args.lr /10
       train(epoch)
       test(epoch)
    writer.close()
