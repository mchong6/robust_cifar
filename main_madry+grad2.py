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
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--k', default=10, type=int, help='learning rate')
parser.add_argument('--lambda_grad', default=1, type=float, help='learning rate')
parser.add_argument('--model', default='batch', help='learning rate')
parser.add_argument('--iter_eps', default=2/255, type=float, help='learning rate')
parser.add_argument('--gpu', default=0, type=int, help='gpu')
parser.add_argument('--max_eps', default=8/255, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--warm', '-w', action='store_true', help='resume from checkpoint')
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
# writer = SummaryWriter("logs/test2")

# Model
MEAN = torch.Tensor([0.4914, 0.4822, 0.4465])
STD = torch.Tensor([0.2023, 0.1994, 0.2010])
norm = NormalizeByChannelMeanStd(mean=MEAN, std=STD)

print('==> Building model..')
if args.model =='batch':
    net = torch.nn.DataParallel(nn.Sequential(norm, ResNet18()).cuda())
else:
    net = torch.nn.DataParallel(nn.Sequential(norm, ResNet18_bn()).cuda())

cudnn.benchmark = True

if args.resume or args.test:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/%s_%d_grad%d_lambda_%.1f.t7'%(args.model, args.adv, args.grad, args.lambda_grad))
#     checkpoint = torch.load('./checkpoint/gn_ckpt_adv0_grad1_lambda_100.0.t7')
    net.load_state_dict(checkpoint['net'])
#     best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer = optim.Adam(net.parameters(), lr=0.01)

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

# def calc_grad(inputs, targets, classifier):
#     inputs.requires_grad_()
#     with torch.enable_grad():
#         output = classifier(inputs)
#         loss = criterion(output, targets)
# #         cw_loss = cw(output, targets)
        
#     grad = torch.autograd.grad(loss, [inputs], create_graph=True)[0]
#     grad_loss = grad.norm(2)
#     return loss, args.lambda_grad*grad_loss, output

def calc_grad(adv_im, inputs, targets, classifier):
    inputs.requires_grad_()
    real_output = classifier(inputs)
    real_loss = criterion(real_output, targets)
    gradients = torch.autograd.grad(outputs=real_loss, inputs=inputs,
                  grad_outputs=torch.ones(real_loss.size()).cuda(),
                  create_graph=True, retain_graph=True)[0]
    gradients = gradients.view(inputs.size(0), -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12).mean()
    
    adv_output = classifier(adv_im)
    adv_loss = criterion(adv_output, targets)
        
    return real_loss+adv_loss, args.lambda_grad*gradients_norm, adv_output

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
#         inputs += torch.zeros_like(inputs).uniform_(-8/255, 8/255)
        
        adv_im = adversary.perturb(inputs, targets).detach()
        loss, grad_loss, outputs = calc_grad(adv_im, inputs, targets, net)
        optimizer.zero_grad()
        (loss+grad_loss).backward()
        optimizer.step()
        
#         writer.add_scalar("loss", loss.item(), len(trainloader)*epoch+batch_idx)
#         writer.add_scalar("grad loss", grad_loss.item(), len(trainloader)*epoch+batch_idx)

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
#         torch.save(state, './checkpoint/madry+grad_lambda%f.t7'%args.lambda_grad)
#         torch.save(state, './checkpoint/ckpt_grad_%.3f.t7'%args.lambda_grad)
        torch.save(state, './checkpoint/real+grad+adv.t7')
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
#     writer.close()
