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
from wideresnet import WideResNet_feat
from utils import progress_bar
from advertorch.attacks import LinfPGDAttack
from advertorch.utils import NormalizeByChannelMeanStd
from advertorch.context import ctx_noparamgrad_and_eval
# from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--k', default=10, type=int, help='learning rate')
parser.add_argument('--lambda_MI', default=1, type=float, help='learning rate')
parser.add_argument('--model', default='batch', help='learning rate')
parser.add_argument('--iter_eps', default=2/255, type=float, help='learning rate')
parser.add_argument('--gpu', default=0, type=int, help='gpu')
parser.add_argument('--max_eps', default=8/255, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--warm', '-w', action='store_true', help='resume from checkpoint')
parser.add_argument('--test', '-t', action='store_true', help='resume from checkpoint')
parser.add_argument('--switch', '-s', action='store_true', help='do adversarial training')
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

class Mine(nn.Module):
    def __init__(self, input_size, hidden_size=256):
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
#         print(input.min(), input.max())
        return self.fc1(input)

def mutual_information(joint, marginal, mine_net, ma_et=1., ma_rate=0.1):
#    t = mine_net(joint)
#    et = torch.exp(mine_net(marginal))
#    mi_lb = torch.mean(t) - torch.log(torch.mean(et).clamp(min=1e-8))
#    ma_et = (1-ma_rate)*ma_et + ma_rate*torch.mean(et)
#    
#    # unbiasing use moving average
#    loss = -(torch.mean(t) - (1/(ma_et.mean()+1e-8)).detach()*torch.mean(et))
##     loss = -mi_lb
#    return loss, ma_et
    Ej = -F.softplus(-mine_net(joint)).mean()
    Em = F.softplus(mine_net(marginal)).mean()
    LOCAL = (Em - Ej)
    return LOCAL, 0

# Model
MEAN = torch.Tensor([0.4914, 0.4822, 0.4465])
STD = torch.Tensor([0.2023, 0.1994, 0.2010])
norm = NormalizeByChannelMeanStd(mean=MEAN, std=STD)

print('==> Building model..')
if args.model =='batch':
    net = torch.nn.DataParallel(nn.Sequential(norm, ResNet18_feat()).cuda())
elif args.model =='group':
    net = torch.nn.DataParallel(nn.Sequential(norm, ResNet18_bn()).cuda())
elif args.model =='wide':
<<<<<<< HEAD
    net = torch.nn.DataParallel(nn.Sequential(norm, WideResNet_feat()).cuda())

mine_net = torch.nn.DataParallel(Mine(1280)).cuda()
=======
    net = torch.nn.DataParallel(nn.Sequential(norm, WideResNet()).cuda())

mine_net = torch.nn.DataParallel(Mine(1024)).cuda()
>>>>>>> 90ca75f8b20240c8f9da0d78a425fbf73ddb31ae

cudnn.benchmark = True
ma_et=1

if args.resume or args.test:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
<<<<<<< HEAD
    checkpoint = torch.load('./checkpoint/wide_MI_s0_lambda_1.0.t7')
=======
    checkpoint = torch.load('./checkpoint/batch_MI_s_1_lambda_0.1.t7')
>>>>>>> 90ca75f8b20240c8f9da0d78a425fbf73ddb31ae
    net.load_state_dict(checkpoint['net'])
    mine_net.load_state_dict(checkpoint['mine'])
#     best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer = optim.Adam(net.parameters(), lr=args.lr)
mine_optimizer = optim.Adam(mine_net.parameters(), lr=0.01)

adversary = LinfPGDAttack(
    net, eps=8/255, eps_iter=2/255, nb_iter=7,
    rand_init=True, targeted=False)


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
<<<<<<< HEAD
    if epoch >= 100:
        lr = args.lr * 0.1
    if epoch >= 150:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def BIM(image, classifier, target, eps, itr_eps=2 / 255, itr=7):
    classifier.eval()
    origin = image.clone()
    for _ in range(itr):
        image.requires_grad = True
        out_image = image
        with torch.enable_grad():
            _, output = classifier(out_image)
            loss = F.cross_entropy(output, target)
        grad = torch.autograd.grad(loss, [image])[0]
        image = image.detach() + itr_eps * torch.sign(grad.detach())
        image = torch.min(torch.max(image, origin - eps), origin + eps)
        image = image.clamp(0, 1).detach()
    classifier.train()
    return image.detach()

=======
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

>>>>>>> 90ca75f8b20240c8f9da0d78a425fbf73ddb31ae
def calc_grad(model,
                mine,
                x_natural,
                y,
                ma_et,
                optimizer,
                step_size=2/255,
                epsilon=8/255,
                perturb_steps=7):
    # define KL-loss
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    with torch.no_grad():
        feat_natural, logits_natural = model(x_natural)
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            feat_adv, logits_adv = model(x_adv)
            if args.switch:
                loss = F.cross_entropy(logits_adv, y)
            else:
                joint = torch.cat([feat_natural, feat_adv], 1)
                perm = torch.randperm(x_adv.size(0))
                marginal = torch.cat([feat_natural, feat_adv[perm]], 1)
                loss, ma_et = mutual_information(joint, marginal, mine, ma_et)
            
        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
    model.train()

    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    feat_natural, logits_natural = model(x_natural)
    feat_adv, logits_adv = model(x_adv)
    loss_natural = F.cross_entropy(logits_natural, y)
    
<<<<<<< HEAD
=======
    joint = torch.cat([feat_natural, feat_adv], 1)
    marginal = torch.cat([feat_natural, feat_adv[torch.randperm(feat_adv.size(0))]], 1)
    MI_loss, ma_et = mutual_information(joint, marginal, mine, ma_et)
>>>>>>> 90ca75f8b20240c8f9da0d78a425fbf73ddb31ae
    
    loss = loss_natural 
    MI_loss = args.lambda_MI * MI_loss 
    return loss, MI_loss, logits_adv

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
<<<<<<< HEAD
    mine_net.train()
=======
>>>>>>> 90ca75f8b20240c8f9da0d78a425fbf73ddb31ae
    total_train_loss = 0
    total_robust_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
<<<<<<< HEAD
        adv_inputs = BIM(inputs, net, targets, 8/255).detach()

        feat_natural, output_natural = net(inputs)
        feat_adv, output_adv = net(adv_inputs)

        joint = torch.cat([feat_natural, feat_adv], 1)
        marginal = torch.cat([feat_natural, feat_adv[torch.randperm(feat_adv.size(0))]], 1)
        MI_loss, ma_et = mutual_information(joint, marginal, mine_net, 1)

        MI_loss = args.lambda_MI * MI_loss
        loss = criterion(torch.cat([output_natural, output_adv], 0), torch.cat([targets, targets],0))
        total_loss = loss + MI_loss

=======
#         inputs += torch.zeros_like(inputs).uniform_(-8/255, 8/255)
        
        loss, robust_loss, outputs = calc_grad(net, mine_net, inputs, targets, ma_et, optimizer)
        total_loss = loss + robust_loss
>>>>>>> 90ca75f8b20240c8f9da0d78a425fbf73ddb31ae
        optimizer.zero_grad()
        mine_optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        mine_optimizer.step()
        
<<<<<<< HEAD
        total_train_loss += loss.item()
        total_robust_loss += MI_loss.item()
        
        _, predicted = output_adv.max(1)
=======
#         writer.add_scalar("loss", loss.item(), len(trainloader)*epoch+batch_idx)
#         writer.add_scalar("grad loss", grad_loss.item(), len(trainloader)*epoch+batch_idx)

        total_train_loss += loss.item()
        total_robust_loss += robust_loss.item()
        
        _, predicted = outputs.max(1)
>>>>>>> 90ca75f8b20240c8f9da0d78a425fbf73ddb31ae
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Robust Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (total_train_loss/(batch_idx+1), total_robust_loss/(batch_idx+1),100.*correct/total, correct, total))

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
            'mine': mine_net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/%s_MI_s%d_lambda_%.1f.t7'%(args.model, args.switch, args.lambda_MI))
        best_acc = acc


if args.test:
    test(0)
else:
<<<<<<< HEAD
    max_epoch = 200
=======
    max_epoch = 100
>>>>>>> 90ca75f8b20240c8f9da0d78a425fbf73ddb31ae
    for epoch in range(start_epoch, max_epoch):
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
        test(epoch)
#     writer.close()
