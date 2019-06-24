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
parser.add_argument('--alpha', default=1, type=float, help='learning rate')
parser.add_argument('--model', default='batch', help='learning rate')
parser.add_argument('--iter_eps', default=2/255, type=float, help='learning rate')
parser.add_argument('--gpu', default=0, type=int, help='gpu')
parser.add_argument('--max_eps', default=8/255, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--test', '-t', action='store_true', help='resume from checkpoint')
parser.add_argument('--train_both', '-b', action='store_true', help='do adversarial training')
args = parser.parse_args()
print(args)
import torch
import torch.nn as nn

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

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
net = torch.nn.DataParallel(nn.Sequential(norm, ResNet18_feat()).cuda())
center_mod = CenterLoss(num_classes=10, feat_dim=512, use_gpu=True)

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
criterion_kl = nn.KLDivLoss(size_average=False)
optimizer = optim.SGD(list(net.parameters()) + list(center_mod.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)

def BIM(image, classifier, center_mod, target, eps, itr_eps=2 / 255, itr=10):
    origin = image.clone()
    
    centers = center_mod.centers[target]
    for _ in range(itr):
        image.requires_grad = True
        out_image = image
        with torch.enable_grad():
            feat, output = classifier(out_image)
            center_loss = F.mse_loss(feat[-1], centers)
            
        grad = torch.autograd.grad(center_loss, [image])[0]
        image = image.detach() + itr_eps * torch.sign(grad.detach())
        image = torch.min(torch.max(image, origin - eps), origin + eps)
        image = image.clamp(0, 1).detach()

    return image.detach()

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    center_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        bs = inputs.size(0)
        
        net.eval()
        adv_im = BIM(inputs, net, center_mod, targets, 8/255, itr_eps=2 / 255, itr=7).detach()
        net.train()
            
        if args.train_both:
            all_im = torch.cat([inputs, adv_im], 0)
            targets = torch.cat([targets,targets], 0)
        else:
            all_im = adv_im
           
        feat, outputs = net(all_im)
        center_loss = args.alpha*center_mod(feat[-1], targets) 
        xe_loss = criterion(outputs, targets)
        loss = center_loss + xe_loss
        
        optimizer.zero_grad()
        loss.backward()
        
        for param in center_mod.parameters():
            # lr_cent is learning rate for center loss, e.g. lr_cent = 0.5
            param.grad.data *= (0.5 / (args.alpha * get_lr(optimizer)))
        optimizer.step()

        train_loss += xe_loss.item()
        center_loss += center_loss.item()
        if args.train_both:
            outputs = outputs[bs:]
            targets = targets[bs:]
            
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Grad Loss: %.6f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), center_loss/(batch_idx+1), 100.*correct/total, correct, total))

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
#         torch.save(state, './checkpoint/madry+grad_lambda%f.t7'%args.lambda_grad)
#         torch.save(state, './checkpoint/ckpt_grad_%.3f.t7'%args.lambda_grad)
        torch.save(state, './checkpoint/both_center_lambda%.1f.t7'%args.alpha)
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
