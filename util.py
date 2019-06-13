from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
from io import BytesIO

import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.utils import data

import numbers
import math
import matplotlib.pyplot as plt


import pickle
import json
import logging
import shutil
from tqdm import tqdm_notebook

import dominate
from dominate.tags import meta, h3, table, tr, td, p, a, img, br, div


def random_noise(im, type='uniform',p=0.5, eps=4 / 255, smooth=False, sigma=1):
    """
    Adds a Gaussian noise to an input image with a chance of p
 
    Args:
        im: (Tensor) Image of size [B, C, H, W]
        p: (float) Probability of adding noise
        eps: (float) std of Gaussian noise added
    """
    random_p = torch.zeros(im.size(0)).bernoulli_(p=p).view(-1, 1, 1, 1)

    if type == 'gaussian':
        noise = torch.zeros_like(im).normal_(0, eps)
    elif type == 'uniform':
        noise = torch.zeros_like(im).uniform_(-eps, eps)
    
    if im.is_cuda:
        random_p = random_p.cuda()
        noise = noise.cuda()
    
    if smooth:
        noise = rescale(gaussian_smooth(noise, 0, sigma), eps)

    im = im + random_p * noise
    return im.clamp(0, 1)


class InfiniteSampler(data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        i = 0
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                np.random.seed()
                order = np.random.permutation(self.num_samples)
                i = 0


def display_image(image, unnorm=False, title=''):
    # image is [3,h,w] or [1,3,h,w] tensor [0,1]
    if image.is_cuda:
        image = image.cpu()
    if image.dim() == 4:
        image = image[0]
    if unnorm:
        image = unnormalize(image.unsqueeze(0)).squeeze(0)

    image = image.clamp(0, 1).permute(1, 2, 0).detach().numpy()
    plt.figure()
    plt.title(title)
    plt.axis('off')
    plt.imshow(image)


def find_lr(
    model, optimizer, dataloader, params, init_value=1e-8, final_value=10.0, beta=0.98
):
    model.module.denoise_model.train()
    # num = len(dataloader)-1
    num = 199
    mult = (final_value / init_value) ** (1 / num)
    lr = init_value
    optimizer.param_groups[0]["lr"] = lr
    avg_loss = 0.0
    best_loss = 0.0
    losses = []
    log_lrs = []

    for i, (real_im, fake_im) in enumerate(tqdm_notebook(dataloader)):
        if i > 199:
            break
        batch_num = i + 1
        model.zero_grad()
        real_im = real_im.to(params.device)
        fake_im = fake_im.to(params.device)
        output, topk, all_loss = model(real_im, fake_im)
        loss = 0
        for key, coef in params.lambdas.items():
            if key in all_loss:
                value = coef * all_loss[key].mean()
                loss += value
        # loss = loss[0] # take only L1 loss
        # Compute the smoothed loss
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** batch_num)
        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        # Record the best loss
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss
        # Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        # Do the SGD step
        loss.backward()
        optimizer.step()
        # Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]["lr"] = lr

    return log_lrs, losses


def check_filenames_match(list1, list2):
    """
    Given two lists of filepaths, ensure their filenames without ext are the same
    """
    length1 = len(list1)
    length2 = len(list2)
    if length1 != length2:
        raise Exception("length of lists %d, %d are not the same"%(length1, length2))

    for i in range(len(list1)):
        name1 = get_filename(list1[i])
        name2 = get_filename(list2[i])
        if name1 != name2:
            raise Exception("%s and %s are not the same" % (name1, name2))


def get_filename(filepath):
    """
    Get filename without extensions given path to file
    Args:
        filepath: Path to file
    """
    filename = os.path.splitext(os.path.basename(filepath))[0]
    return filename


def list_relative_paths(directory, exts=()):
    """
    Get relative paths of all files with given extensions in the directory
    Args:
        directory: Path to list all files
        exts (tuple): tuple of all extensions to list
    """
    paths = []
    for path in os.listdir(directory):
        if not len(exts) or path.endswith(exts):
            paths.append(os.path.join(directory, path))
    return sorted(paths)


def create_dirs(directories):
    """
    Create directories if it does not exist
    Args:
        directory: Path to directory to be created
    """
    if isinstance(directories, list):
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    else:
        os.makedirs(directories, exist_ok=True)


def setup_json(parameters, directory):
    """
    Dump parameters into json file in directory
    Args:
        parameters: Parameters of the experiment
        directory: Path to save json
    Output:
        Params
    """
    json_path = os.path.join(directory, "params.json")

    # Write parameters to json file
    with open(json_path, "w") as f:
        json.dump(parameters, f, indent=4)


class Params:
    """Class that loads hyperparameters from a json file or dictionary.

    Example:
    ```
    params = Params(dict)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, parameters):
        self.__dict__.update(parameters)

    def save(self, json_path):
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage:
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, "w") as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


MEAN = torch.Tensor([0.485, 0.456, 0.406])
STD = torch.Tensor([0.229, 0.224, 0.225])


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


def save_checkpoint(state, is_best, checkpoint, name="model"):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'.
    If is_best==True, also saves checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, name + "_" + "last.pth.tar")
    if not os.path.exists(checkpoint):
        os.mkdir(checkpoint)

    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, name + "_" + "best.pth.tar"))


def load_checkpoint(checkpoint, model, optimizer=None, keys=[]):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
        keys: (str) Keys to load from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))

    checkpoint_dict = torch.load(checkpoint)
    try:
        model.load_state_dict(checkpoint_dict["state_dict"])
    except:
        flexible_load_state(model, checkpoint_dict["state_dict"])

    if optimizer:
        optimizer.load_state_dict(checkpoint_dict["optim_dict"])

    if len(keys):
        dict = {}
        for key in keys:
            if key in checkpoint_dict:
                dict[key] = checkpoint_dict[key]
            else:
                dict[key] = None
        return dict


def flexible_load_state(model, checkpoint_dict):
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    checkpoint_dict = {k: v for k, v in checkpoint_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(checkpoint_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)


class SGDR:
    """
    Creates a learning rate scheduler that uses a cosine annealing schedule with warm restarts
   
    Parameters
    ----------
    data_sz : (int) Size of dataloader
    cycle_len: (int) Decrease LR over how many epochs
    cycle_mult: (int) Length of decrease multiplier (Decrease over 1 epoch > decrease over 2 > over 4 ...
    optimizer: (Optimizer)
    max_lr: Maximum learning rate to start
    """

    def __init__(self, data_sz, cycle_len, cycle_mult, optimizer, max_lr):
        self.max_i = cycle_len * data_sz
        self.cycle_mult = cycle_mult
        self.optimizer = optimizer
        self.i = 0
        self.max_lr = max_lr
        self.min_lr = 1e-8

    def step(self):
        # i starts from 0, so we have to subtract 1 from max_i.
        # Do not update max_i since mutiplying with -1 will change total cycle length
        cur_lr = (
            self.min_lr
            + (self.max_lr - self.min_lr)
            * (1 + math.cos(math.pi * self.i / (self.max_i - 1)))
            / 2
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = cur_lr

        self.i += 1
        if self.i >= self.max_i:
            self.i = 0
            self.max_i *= self.cycle_mult


class OneCycle(object):
    """
    In paper (https://arxiv.org/pdf/1803.09820.pdf), author suggests to do one cycle during
    whole run with 2 steps of equal length. During first step, increase the learning rate
    from lower learning rate to higher learning rate. And in second step, decrease it from
    higher to lower learning rate. This is Cyclic learning rate policy. Author suggests one
    addition to this. - During last few hundred/thousand iterations of cycle reduce the
    learning rate to 1/100th or 1/1000th of the lower learning rate.
    Also, Author suggests that reducing momentum when learning rate is increasing. So, we make
    one cycle of momentum also with learning rate - Decrease momentum when learning rate is
    increasing and increase momentum when learning rate is decreasing.
    Args:
        num_iter        Total number of iterations including all epochs
        max_lr          The optimum learning rate. This learning rate will be used as highest
                        learning rate. The learning rate will fluctuate between max_lr to
                        max_lr/div and then (max_lr/div)/div.
        momentum_vals   The maximum and minimum momentum values between which momentum will
                        fluctuate during cycle.
                        Default values are (0.95, 0.85)
        prcnt           The percentage of cycle length for which we annihilate learning rate
                        way below the lower learnig rate.
                        The default value is 10
        div             The division factor used to get lower boundary of learning rate. This
                        will be used with max_lr value to decide lower learning rate boundary.
                        This value is also used to decide how much we annihilate the learning
                        rate below lower learning rate.
                        The default value is 10.
        cur_iter        Resume from current iteration
    """

    def __init__(
        self,
        optimizer,
        num_iter,
        max_lr,
        momentum_vals=(0.95, 0.85),
        prcnt=10,
        div=10,
        cur_iter=0,
    ):
        self.optimizer = optimizer
        self.nb = num_iter
        self.div = div
        self.step_len = int(self.nb * (1 - prcnt / 100) / 2)
        self.high_lr = max_lr
        self.low_mom = momentum_vals[1]
        self.high_mom = momentum_vals[0]
        self.prcnt = prcnt
        self.iteration = cur_iter
        self.lrs = []
        self.moms = []

    def step(self):
        self.iteration += 1
        lr = self.calc_lr()
        mom = self.calc_mom()
        self.update_lr(lr)
        self.update_mom(mom)
        return (lr, mom)

    def calc_lr(self):
        if self.iteration == self.nb:
            self.iteration = 0
            self.lrs.append(self.high_lr / self.div)
            return self.high_lr / self.div
        if self.iteration > 2 * self.step_len:
            ratio = (self.iteration - 2 * self.step_len) / (self.nb - 2 * self.step_len)
            lr = self.high_lr * (1 - 0.99 * ratio) / self.div
        elif self.iteration > self.step_len:
            ratio = 1 - (self.iteration - self.step_len) / self.step_len
            lr = self.high_lr * (1 + ratio * (self.div - 1)) / self.div
        else:
            ratio = self.iteration / self.step_len
            lr = self.high_lr * (1 + ratio * (self.div - 1)) / self.div
        self.lrs.append(lr)
        return lr

    def calc_mom(self):
        if self.iteration == self.nb:
            self.iteration = 0
            self.moms.append(self.high_mom)
            return self.high_mom
        if self.iteration > 2 * self.step_len:
            mom = self.high_mom
        elif self.iteration > self.step_len:
            ratio = (self.iteration - self.step_len) / self.step_len
            mom = self.low_mom + ratio * (self.high_mom - self.low_mom)
        else:
            ratio = self.iteration / self.step_len
            mom = self.high_mom - ratio * (self.high_mom - self.low_mom)
        self.moms.append(mom)
        return mom

    def update_lr(self, lr):
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def update_mom(self, mom):
        for g in self.optimizer.param_groups:
            g["momentum"] = mom


class HTML:
    """This HTML class allows us to save images and write texts into a single HTML file.
     It consists of functions such as <add_header> (add a text header to the HTML file),
     <add_images> (add a row of images to the HTML file), and <save> (save the HTML to the disk).
     It is based on Python library 'dominate', a Python library for creating and manipulating HTML documents using a DOM API.
    """

    def __init__(self, web_dir, title, refresh=0):
        """Initialize the HTML classes
        Parameters:
            web_dir (str) -- a directory that stores the webpage. HTML file will be created at <web_dir>/index.html; images will be saved at <web_dir/images/
            title (str)   -- the webpage name
            refresh (int) -- how often the website refresh itself; if 0; no refreshing
        """
        self.title = title
        self.web_dir = web_dir
        self.img_dir = os.path.join(self.web_dir, "images")
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        self.doc = dominate.document(title=title)
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(refresh))

    def get_image_dir(self):
        """Return the directory that stores images"""
        return self.img_dir

    def add_header(self, text):
        """Insert a header to the HTML file
        Parameters:
            text (str) -- the header text
        """
        with self.doc:
            h3(text)

    def add_images(self, ims, txts, links, width=400):
        """add images to the HTML file
        Parameters:
            ims (str list)   -- a list of image paths
            txts (str list)  -- a list of image names shown on the website
            links (str list) --  a list of hyperref links; when you click an image, it will redirect you to a new page
        """
        self.t = table(border=1, style="table-layout: fixed;")  # Insert a table
        self.doc.add(self.t)
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(
                        style="word-wrap: break-word;", halign="center", valign="top"
                    ):
                        div(link)
                        with p():
                            with a(href=os.path.join("images", im)):
                                img(
                                    style="width:%dpx" % width,
                                    src=os.path.join("images", im),
                                )
                            br()
                            p(txt)

    def save(self):
        """save the current content to the HMTL file"""
        html_file = "%s/index.html" % self.web_dir
        f = open(html_file, "wt")
        f.write(self.doc.render())
        f.close()


def save_images(webpage, visuals, aspect_ratio=1.0, width=256):
    """Save images to the disk.
    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these images (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores data in the following format
                                 -- {key: {names: "name.png",
                                           images: image tensor,
                                           labels: label list}}
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width
    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()

    for i in range(len(visuals["real"]["names"])):
        webpage.add_header(visuals["real"]["names"][i])  # webpage name
        ims, txts, links = [], [], []

        # for the same index in all dict keys, view them on the same row
        for key, dict_entry in visuals.items():
            save_path = os.path.join(image_dir, dict_entry["names"][i])

            # create directory paths for images if doesnt exist already
            parent_dir = os.path.dirname(save_path)
            if parent_dir != "":
                create_dirs(parent_dir)

            save_image(dict_entry["images"][i], save_path)
            ims.append(dict_entry["names"][i])
            txts.append(dict_entry["labels"][i])
            links.append(key)
        webpage.add_images(ims, txts, links, width=width)


class_idx = json.load(open("imagenet_class_index.json"))
map_idx2label = [class_idx[str(i)][1] for i in range(len(class_idx))]

def idx2labels(idx, val, k=2):
    """
    Converts a imagenet classes index to labels and confidence

    Args:
        idx: (LongTensor) sorted dim=[1000]
        val: (FloatTensor) sorted dim=[1000]
        k: (Int) Number of labels/confidence to return
    Output:
        labels: (str) Label and confidence. e.g) 'robin: 0.997, coucal: 0.001'
    """
    labels = [
        "%s: %.3f" % (map_idx2label[idx[j]], val[j].item()) for j in range(k)
    ]
    labels = ", ".join(labels)
    return labels
    
    
def logits2labels(logits, k=2):
    """
    Converts logits to labels and confidence

    Args:
        idx: (FloatTensor) dim=[B, 1000]
        k: (Int) Number of labels/confidence to return
    Output:
        labels: (list) List of labels and confidences
    """
    val, idx = F.softmax(logits, 1).sort(dim=1, descending=True)  # [B, 1000]
    all_labels = []

    for i in range(idx.size(0)):
        labels = idx2labels(idx[i], val[i], k)
        all_labels.append(labels)

    return all_labels


with open('cls2idx.pickle', 'rb') as f:
    class2idx_dict = pickle.load(f)

def class2idx(classes):
    """
    Converts a imagenet classes (n02119789) to the class index

    Args:
        classes: (list) list of classes
    Output:
        indices: (LongTensor) class indices
    """
    indices = torch.LongTensor(len(classes))
    for i, class_name in enumerate(classes):
        indices[i] = class2idx_dict[class_name]
    return indices


def attack(image, classifier, target, eps, itr, attack_method):

    if np.random.randint(0, 1):
        image += torch.zeros_like(image).uniform_(-eps, eps).cuda()

    if attack_method == "FGSM":
        adv_image = FGSM(image, classifier, target, eps)

    elif attack_method == "IFGSM":
        adv_image = IFGSM(image, classifier, target, eps, itr=itr)

    elif attack_method == "BIM":
        adv_image = BIM(image, classifier, target, eps, itr=itr)

    return adv_image.detach()


def FGSM(image, classifier, target, eps):
    image.requires_grad_()

    with torch.enable_grad():
        output = classifier(normalize(image))
        loss = F.cross_entropy(
            output, target, size_average=False
        )

    grad = torch.autograd.grad(loss, [image])[0]
    image = image.detach() + eps * torch.sign(grad.detach())
    image = image.clamp(0, 1)
    return image


def BIM(image, classifier, target, eps, itr_eps=1 / 255, itr=30):
    origin = image.clone()
    for _ in range(itr):
        image.requires_grad = True
        with torch.enable_grad():
            output = classifier(normalize(image))
            loss = F.cross_entropy(
                output, target, size_average=False
            )
        grad = torch.autograd.grad(loss, [image])[0]
        image = image.detach() + itr_eps * torch.sign(grad.detach())
        image = torch.min(torch.max(image, origin - eps), origin + eps)
        image = image.clamp(0, 1).detach()
    return image


def IFGSM(image, classifier, target, eps, itr_eps=1 / 255, itr=30):
    origin = image.clone()
    for _ in range(itr):
        image.requires_grad = True
        with torch.enable_grad():
            output = classifier(normalize(image))
            loss = F.cross_entropy(output, target.long(), size_average=False)
        grad = torch.autograd.grad(loss, [image])[0]
        image = image.detach() - itr_eps * torch.sign(grad.detach())
        image = torch.min(torch.max(image, origin - eps), origin + eps)
        image = image.clamp(0, 1).detach()
    return image


def PNG2JPG(im, quality=75):
    with BytesIO() as f:
        im.save(f, format="JPEG", quality=quality)
        return Image.open(BytesIO(f.getvalue()))
    
class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
            
    Usage:
    smoothing = GaussianSmoothing(3, 5, 1)
    input = torch.rand(1, 3, 100, 100)
    input = F.pad(input, (2, 2, 2, 2), mode='reflect')
    output = smoothing(input)
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        """
        If kernel_size = 0, calculate it based on sigma
        Ensure kernel_size is odd
        """
        if 0 in kernel_size:
            kernel_size = []
            for x in sigma:
                k = int(x*4+1)
                if k % 2 == 0:
                    k +=1 
                kernel_size.append(k)

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        
        # Calculate pad size to ensure output is same dim
        self.padding = [int((kernel_size[0]-1)/2)] * (channels+1)

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """

        input = F.pad(input, self.padding, mode='reflect')
        return self.conv(input, weight=self.weight, groups=self.groups)


def gaussian_smooth(im, kernel_size, sigma, dim=2):
    if isinstance(sigma, numbers.Number):
        if sigma == 0:
            return im
        else:
            sigma = [sigma] * dim
    if isinstance(kernel_size, numbers.Number):
        kernel_size = [kernel_size] * dim
    """
    If kernel_size = 0, calculate it based on sigma
    Ensure kernel_size is odd
    """
    if 0 in kernel_size:
        kernel_size = []
        for x in sigma:
            k = int(x*4+1)
            if k % 2 == 0:
                k +=1 
            kernel_size.append(k)

    channels = im.size(1)
    # The gaussian kernel is the product of the
    # gaussian function of each dimension.
    kernel = 1
    meshgrids = torch.meshgrid(
        [
            torch.arange(size, dtype=torch.float32)
            for size in kernel_size
        ]
    )
    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                  torch.exp(-((mgrid - mean) / std) ** 2 / 2)

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)

    # Reshape to depthwise convolutional weight
    kernel = kernel.view(1, 1, *kernel.size())
    kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
    if im.is_cuda:
        kernel = kernel.cuda()

    # Calculate pad size to ensure output is same dim
    padding = [int((kernel_size[0]-1)/2)] * (channels+1)

    if dim == 1:
        conv = F.conv1d
    elif dim == 2:
        conv = F.conv2d
    elif dim == 3:
        conv = F.conv3d
    else:
        raise RuntimeError(
            'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
        )

    im = F.pad(im, padding, mode='reflect')
    return conv(im, weight=kernel, groups=channels)


def rescale(im, eps):
    mean = im.mean()
    im=im-mean
    
    return (2*(im-im.min())/(im.max()-im.min()) - 1) * eps


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

def init_weights(model, init_type='kaiming', gain=0.02):
    '''
    initialize network's weights
    init_type: normal | xavier | kaiming | orthogonal
    '''

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)

            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    model.apply(init_func)

class softCrossEntropy(nn.Module):
    def __init__(self):
        super(softCrossEntropy, self).__init__()
        return

    def forward(self, inputs, target):
        """
        :param inputs: predictions
        :param target: target labels
        :return: loss
        """
        return torch.mean(torch.sum(-target * F.log_softmax(inputs, dim=1), dim=1))
