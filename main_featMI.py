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
import argparse

from models import *
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--lambda_MI', default=1, type=float, help='learning rate')
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
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
class Mine(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_size, 1)
            )

    def forward(self, input):
        return self.fc1(input)


# Model
print('==> Building model..')
net = ResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    mine_net = torch.nn.DataParallel(Mine(10))
    cudnn.benchmark = True
    print('true')

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
mine_optimizer = optim.Adam(mine_net.parameters(), lr=1e-3)

def swap_perm(source):
    # [b x k]
    output = source.clone()
    for i in range(output.size(1)):
        output[:, i] = output[torch.randperm(output.size(0)), i]
    return output

def mutual_information(joint, marginal, mine_net):
    Ej = -F.softplus(-mine_net(joint)).mean()
    Em = F.softplus(mine_net(marginal)).mean()
    LOCAL = (Em - Ej)
    return LOCAL

def inference(img, net, mine_net):
    output = net(img)
    joint = F.tanh(output)
    marginal = swap_perm(joint)
        
    MI_loss = mutual_information(joint, marginal, mine_net)
    return output, MI_loss

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    total_train_loss = 0
    total_MI_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        _, MI_loss = inference(inputs, net, mine_net)
        mine_optimizer.zero_grad()
        MI_loss.backward()
        mine_optimizer.step()
        
        outputs, MI_loss = inference(inputs, net, mine_net)
        xe_loss = criterion(outputs, targets)
        
        net_loss = xe_loss - args.lambda_MI * MI_loss
        optimizer.zero_grad()
        net_loss.backward()
        optimizer.step()

        total_train_loss += xe_loss.item()
        total_MI_loss += MI_loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | MI Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (total_train_loss/(batch_idx+1), total_MI_loss/(batch_idx+1),100.*correct/total, correct, total))

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
            loss = F.cross_entropy(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_featMI.t7')
        best_acc = acc


# test(0)
for epoch in range(start_epoch, start_epoch+200):
    if epoch == 100:
        for g in optimizer.param_groups:
            g["lr"] = args.lr /10
    train(epoch)
    test(epoch)
