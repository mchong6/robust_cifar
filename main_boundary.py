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
import argparse

from models import *
from utils import progress_bar
from advertorch.attacks import LinfPGDAttack
from advertorch.utils import NormalizeByChannelMeanStd
from advertorch.context import ctx_noparamgrad_and_eval
# from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--k', default=10, type=int, help='learning rate')
parser.add_argument('--lambda_grad', default=1, type=float, help='learning rate')
parser.add_argument('--iter_eps', default=2/255, type=float, help='learning rate')
parser.add_argument('--gpu', default=0, type=int, help='gpu')
parser.add_argument('--max_eps', default=8/255, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--test', '-t', action='store_true', help='resume from checkpoint')
parser.add_argument('--adv', '-a', action='store_true', help='do adversarial training')
parser.add_argument('--grad', '-g', action='store_true', help='do adversarial training')
args = parser.parse_args()
print(args)

# os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

os.makedirs('logs', exist_ok=True)

# Model
MEAN = torch.Tensor([0.4914, 0.4822, 0.4465])
STD = torch.Tensor([0.2023, 0.1994, 0.2010])
norm = NormalizeByChannelMeanStd(mean=MEAN, std=STD)

print('==> Building model..')
net = torch.nn.DataParallel(nn.Sequential(norm, ResNet18_feat()).cuda())

cudnn.benchmark = True

checkpoint = torch.load('./checkpoint/batch_adv0_grad0_lambda_1.0.t7')
net.load_state_dict(checkpoint['net'])
#freeze last layer
# net.module[-1].linear.weight.requires_grad = False
# net.module[-1].linear.bias.requires_grad = False

if args.resume or args.test:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/boundary.t7')
    net.load_state_dict(checkpoint['net'])
#     best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# optimizer = optim.Adam(net.parameters(), lr=0.01)


def BIM(image, classifier, target, eps, itr_eps=1 / 255, itr=10):
    origin = image.clone()
    for _ in range(itr):
        image.requires_grad = True
        out_image = image
        with torch.enable_grad():
            _, output = classifier(out_image)
            loss = criterion(output, target)
        grad = torch.autograd.grad(loss, [image])[0]
        image = image.detach() + itr_eps * torch.sign(grad.detach())
        image = torch.min(torch.max(image, origin - eps), origin + eps)
        image = image.clamp(0, 1).detach()
    return image.detach()

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    feat_loss = 0
    correct = 0
    total = 0
    for batch_idx, (im, targets) in enumerate(trainloader):
        im, targets = im.to(device), targets.to(device)
        batch_size = im.size(0)
        
        net.eval()
        adv_im = BIM(im, net, targets, 8/255, itr_eps=2 / 255, itr=7).detach()
        net.train()
        
        all_im = torch.cat([im, adv_im], 0)
        feat, outputs = net(all_im)
        real_feat = feat[-1][:batch_size]
        adv_feat = feat[-1][batch_size:]
        
        #move feat further from adv feat
        diff = adv_feat - real_feat
        target_feat = real_feat - 0.3*diff
        
        ce_loss = criterion(outputs[:batch_size], targets)
        feat_loss = 5*F.mse_loss(real_feat, target_feat.detach())
        
        optimizer.zero_grad()
        (ce_loss+feat_loss).backward()
        optimizer.step()
        
        train_loss += ce_loss.item()
        feat_loss += feat_loss.item()
        _, predicted = outputs[batch_size:].max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | feat Loss: %.6f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), feat_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            _, outputs = net(inputs)
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
        torch.save(state, './checkpoint/boundary.t7')
        best_acc = acc


if args.test:
    test(0)
else:
    max_epoch = 200
    for epoch in range(start_epoch, max_epoch):
       if epoch in (100, 150):
           for g in optimizer.param_groups:
               g["lr"]/=10
       train(epoch)
       test(epoch)
