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
from wideresnet import WideResNet
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
elif args.model =='group':
    net = torch.nn.DataParallel(nn.Sequential(norm, ResNet18_bn()).cuda())
elif args.model =='wide':
    net = torch.nn.DataParallel(nn.Sequential(norm, WideResNet()).cuda())

cudnn.benchmark = True

if args.resume or args.test:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/%s_adv%d_grad%d_lambda_%.1f.t7'%(args.model, args.adv, args.grad, args.lambda_grad))
    net.load_state_dict(checkpoint['net'])
#     best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# optimizer = optim.Adam(net.parameters(), lr=0.01)

adversary = LinfPGDAttack(
    net, eps=8/255, eps_iter=2/255, nb_iter=7,
    rand_init=True, targeted=False)

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=2/255,
                epsilon=8/255,
                perturb_steps=7,
                beta=1.0):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    with torch.no_grad:
        logits_natural = model(x_natural)
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                   F.softmax(logits_natural, dim=1))
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    x_natural.requires_grad_()
    logits = model(x_natural)
    adv_logits = model(x_adv)
    loss_natural = F.cross_entropy(logits, y)
    if args.grad:
        gradients = torch.autograd.grad(outputs=loss_natural, inputs=x_natural,
                      grad_outputs=torch.ones(loss_natural.size()).cuda(),
                      create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(inputs.size(0), -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12).mean()
    else:
        gradients_norm = torch.zeros_like(loss_natural).cuda()
        
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(adv_logits, dim=1),
                                                    F.softmax(logits, dim=1))
    loss = loss_natural 
    loss_robust = beta * loss_robust 
    loss_grad = args.lambda_grad * gradients_norm
    return loss, loss_robust, loss_grad, adv_logits

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    total_train_loss = 0
    total_grad_loss = 0
    total_robust_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
#         inputs += torch.zeros_like(inputs).uniform_(-8/255, 8/255)
        
        loss, grad_loss, robust_loss, outputs = calc_grad(train_im, targets, net)
        optimizer.zero_grad()
        (loss+grad_loss+robust_loss).backward()
        optimizer.step()
        
#         writer.add_scalar("loss", loss.item(), len(trainloader)*epoch+batch_idx)
#         writer.add_scalar("grad loss", grad_loss.item(), len(trainloader)*epoch+batch_idx)

        total_train_loss += loss.item()
        total_grad_loss += grad_loss.item()
        total_rob ust_loss += robust_loss.item()
        
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Robust Loss: %.3f | Grad Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (total_train_loss/(batch_idx+1), total_robust_loss/(batch_idx+1), total_grad_loss/(batch_idx+1), 100.*correct/total, correct, total))

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
        torch.save(state, './checkpoint/%s_adv%d_grad%d_lambda_%.1f.t7'%(args.model, args.adv, args.grad, args.lambda_grad))
        best_acc = acc


if args.test:
    test(0)
else:
    max_epoch = 100
    for epoch in range(start_epoch, max_epoch):
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
        test(epoch)
#     writer.close()
