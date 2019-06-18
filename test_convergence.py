import os
from glob import glob
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import torch
import torch.nn as nn

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision
import torch.utils.data as Data
import torchvision.transforms as transforms

from models import *

import matplotlib.pyplot as plt
cudnn.benchmark = True

# from tensorboard import notebook
from advertorch.attacks import LinfPGDAttack
from advertorch.attacks import L2PGDAttack
from advertorch.utils import NormalizeByChannelMeanStd


# In[29]:


MEAN = torch.Tensor([0.4914, 0.4822, 0.4465])
STD = torch.Tensor([0.2023, 0.1994, 0.2010])

def BIM(image, classifier, target, eps, itr_eps=1 / 255, itr=10):
    origin = image.clone()
    for _ in range(itr):
        image.requires_grad = True
        out_image = image
        with torch.enable_grad():
            output = classifier(out_image)
            loss = criterion(output, target)
        grad = torch.autograd.grad(loss, [image])[0]
        image = image.detach() + itr_eps * torch.sign(grad.detach())
        image = torch.min(torch.max(image, origin - eps), origin + eps)
        image = image.clamp(0, 1).detach()
    return image.detach()


transform_test = transforms.Compose([
    transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

norm = NormalizeByChannelMeanStd(
    mean=MEAN, std=STD).cuda()

classifier = torch.nn.DataParallel(nn.Sequential(norm, ResNet18())).cuda().eval()
# classifier = nn.DataParallel(ResNet18())
# checkpoint = torch.load('./checkpoint/madry_baseline.t7')
# checkpoint = torch.load('./checkpoint/madry+grad_lambda1.t7')
# checkpoint = torch.load('./checkpoint/madry+grad_lambda0.100000.t7')
# checkpoint = torch.load('./checkpoint/madry+grad_lambda10.000000.t7')
# checkpoint = torch.load('./checkpoint/ckpt_robust_data_robust_mod.t7')
#checkpoint = torch.load('./checkpoint/ckpt_adv0_lambda_300.000.t7')
#checkpoint = torch.load('./checkpoint/ckpt_adv0_grad1_cw1_lambda_1.0.t7')
#checkpoint = torch.load('./checkpoint/ckpt_adv1_grad1_cw0_lambda_1.0.t7')
#checkpoint = torch.load('./checkpoint/ckpt_adv0_grad1_cw0_lambda_200.0.t7')
checkpoint = torch.load('./checkpoint/ckpt_adv1_grad0_cw0_lambda_1.0.t7')
classifier.load_state_dict(checkpoint['net'])

classifier_norm = nn.Sequential(norm, ResNet18()).cuda().eval()
criterion = nn.CrossEntropyLoss()


from advertorch.attacks import L2PGDAttack

max_eps = 0.25
adversary = L2PGDAttack(
    classifier, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=max_eps,
    nb_iter=20, eps_iter=max_eps/5, rand_init=True, clip_min=0.0, clip_max=1.0,
    targeted=False)

# adv_untargeted = adversary.perturb(cln_data, true_label)

real_correct = 0
fake_correct = 0
total = 0
for itr, (real_im, target) in enumerate((testloader)):
    real_im, target = real_im.cuda(), target.cuda()
    
#     fake_im = robustify(real_im, adv_model, target, 50)
#     fake_im = sparse_attack(real_im, target, adv_model, 8/255, 1/255, 200)
    fake_im = BIM(real_im, classifier, target, 8/255, itr_eps=1/255, itr=20)
#     fake_im = adversary.perturb(real_im, target)
#     fake_im = pgd_lbfgs(real_im, classifier, target, max_eps=8/255, itr_eps=2 / 255, itr=5)
    
    with torch.no_grad():
        real_outputs = classifier(real_im)
        fake_outputs = classifier(fake_im)
        _, real_predicted = real_outputs.max(1)
        _, fake_predicted = fake_outputs.max(1)
        total += target.size(0)
        real_correct += real_predicted.eq(target).sum().item()
        fake_correct += fake_predicted.eq(target).sum().item()
#         print(correct/total)

print(fake_correct/total)
print(real_correct/total)

