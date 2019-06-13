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
os.environ["CUDA_VISIBLE_DEVICES"]="6"
import argparse

from models import *
from utils import progress_bar
from advertorch.utils import NormalizeByChannelMeanStd



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
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

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']

MEAN = torch.Tensor([0.4914, 0.4822, 0.4465])
STD = torch.Tensor([0.2023, 0.1994, 0.2010])

norm = NormalizeByChannelMeanStd(
    mean=MEAN, std=STD)

# net = nn.Sequential(norm, net).cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# optimizer = optim.Adam(net.parameters(), lr=1e-2)


def cw(output, targets):
    y_onehot = torch.FloatTensor(targets.size(0), 10).cuda().zero_()
    y_onehot.scatter_(1, targets.unsqueeze(1), 1)
    # TODO: move this out of the class and make this the default loss_fn
    #   after having targeted tests implemented
    real = (y_onehot * output).sum(dim=1)

    # TODO: make loss modular, write a loss class
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
        cw_loss = cw(output, targets)
        
    grad = torch.autograd.grad(cw_loss, [inputs], create_graph=True)[0]
#     grad_loss = grad.norm(1)
    grad_loss = grad.norm(2)
#     return loss, (-5e-1*grad_loss).clamp(min=-1), output
    return loss, 5e-1*grad_loss, output
#     return loss, 7e2*grad_loss, output
#     grad_loss = grad_loss / grad_loss.detach() * loss.detach() / 10
#     print(loss, grad_loss
#     return loss, grad_loss, output

# def calc_grad(inputs, targets, classifier):
#     inputs.requires_grad_()
#     with torch.enable_grad():
#         output = classifier(inputs)
#         predicted = output.argmax(1)
#         sm = F.softmax(output)
#         vals = sm[torch.arange(predicted.size(0)).cuda(), predicted]
        
#         mask = predicted.eq(targets).float() * (vals > 0.5).float()
#         loss = criterion(output, targets)
        
#     grad = torch.autograd.grad(loss, [inputs], create_graph=True)[0]
#     grad_loss = grad.norm(2, dim=(1,2,3))
#     grad_loss = (mask * grad_loss).mean()
#     return loss, 5e3*grad_loss, output

def mixup_data(x, y, alpha=0.5):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    
    y_onehot = torch.FloatTensor(x.size(0), 10).cuda()
    y_onehot.zero_()
    y_onehot.scatter_(1, y.unsqueeze(1), 1)

    mixed_x = (1-alpha) * x + (alpha) * x[index, :]
    mixed_y = (1-alpha) * y_onehot + alpha * y_onehot[index]
    return mixed_x, mixed_y

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
        inputs += torch.zeros_like(inputs).uniform_(-0.08, 0.08)
#         inputs, targets = mixup_data(inputs, targets)
        optimizer.zero_grad()
        if epoch > -1:
            loss, grad_loss, outputs = calc_grad(inputs, targets, net)
        else:
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            grad_loss = torch.zeros([1]).cuda()
        (loss+grad_loss).backward()
        optimizer.step()

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
            loss = F.cross_entropy(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
#     if acc > best_acc:
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt_grad_cw.t7')
    best_acc = acc


# test(0)
for epoch in range(start_epoch, start_epoch+200):
    if epoch >= 100:
        for g in optimizer.param_groups:
            g["lr"] = args.lr /10
    train(epoch)
    test(epoch)
