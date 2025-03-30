import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.nn.modules.loss import _WeightedLoss
from torchvision import datasets, transforms
from data import transform_imagenet, transform_cifar, transform_svhn, transform_mnist, transform_fashion
from data import TensorDataset, ImageFolder, save_img, img_denormlaize
from data import ClassDataLoader, ClassMemDataLoader, MultiEpochsDataLoader
from data import MEANS, STDS
from train import define_model, train_epoch
from test import test_data, load_ckpt
from misc.augment import DiffAug
from misc import utils
from math import ceil
import glob

from tqdm import tqdm
from tqdm.contrib import tzip
import matplotlib.pyplot as plt
import matplotlib
import random

import models.convnet as CN
import torch.nn.utils.prune as prune

class Synthesizer():
    """Condensed data class
    """
    def __init__(self, args, nclass, nchannel, hs, ws, soft_label = None, device='cuda'):
        self.ipc = args.ipc
        self.nclass = nclass
        self.nchannel = nchannel
        self.size = (hs, ws)
        self.device = device

        self.data = torch.randn(size=(self.nclass * self.ipc, self.nchannel, hs, ws),
                                dtype=torch.float,
                                requires_grad=True,
                                device=self.device)
        self.data.data = torch.clamp(self.data.data / 4 + 0.5, min=0., max=1.)
        self.targets = torch.tensor([np.ones(self.ipc) * i for i in range(nclass)],
                                    dtype=torch.long,
                                    requires_grad=False,
                                    device=self.device).view(-1)

        self.soft_label = soft_label
        self.cls_idx = [[] for _ in range(self.nclass)]
        for i in range(self.data.shape[0]):
            self.cls_idx[self.targets[i]].append(i)

        print("\nDefine synthetic data: ", self.data.shape)

        self.factor = max(1, args.factor)
        self.decode_type = args.decode_type
        self.resize = nn.Upsample(size=self.size, mode='bilinear')
        print(f"Factor: {self.factor} ({self.decode_type})")

    def init(self, loader, init_type='noise'):
        """Condensed data initialization
        """
        if init_type == 'random':
            print("Random initialize synset")
            for c in range(self.nclass):
                img, _ = loader.class_sample(c, self.ipc)
                self.data.data[self.ipc * c:self.ipc * (c + 1)] = img.data.to(self.device)

        elif init_type == 'mix':
            print("Mixed initialize synset")
            for c in range(self.nclass):
                img, _ = loader.class_sample(c, self.ipc * self.factor**2)
                img = img.data.to(self.device)

                s = self.size[0] // self.factor
                remained = self.size[0] % self.factor
                k = 0
                n = self.ipc

                h_loc = 0
                for i in range(self.factor):
                    h_r = s + 1 if i < remained else s
                    w_loc = 0
                    for j in range(self.factor):
                        w_r = s + 1 if j < remained else s
                        img_part = F.interpolate(img[k * n:(k + 1) * n], size=(h_r, w_r))
                        self.data.data[n * c:n * (c + 1), :, h_loc:h_loc + h_r,
                                       w_loc:w_loc + w_r] = img_part
                        w_loc += w_r
                        k += 1
                    h_loc += h_r

        elif init_type == 'noise':
            pass

    def parameters(self):
        parameter_list = [self.data]
        return parameter_list

    def subsample(self, data, target, max_size=-1):
        if (data.shape[0] > max_size) and (max_size > 0):
            indices = np.random.permutation(data.shape[0])
            data = data[indices[:max_size]]
            target = target[indices[:max_size]]

        return data, target

    def decode_zoom(self, img, target, factor):
        """Uniform multi-formation
        """
        h = img.shape[-1]
        remained = h % factor
        if remained > 0:
            img = F.pad(img, pad=(0, factor - remained, 0, factor - remained), value=0.5)
        s_crop = ceil(h / factor)
        n_crop = factor**2

        cropped = []
        for i in range(factor):
            for j in range(factor):
                h_loc = i * s_crop
                w_loc = j * s_crop
                cropped.append(img[:, :, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop])
        cropped = torch.cat(cropped)
        data_dec = self.resize(cropped)
        target_dec = torch.cat([target for _ in range(n_crop)])

        return data_dec, target_dec

    def decode_zoom_multi(self, img, target, factor_max):
        """Multi-scale multi-formation
        """
        data_multi = []
        target_multi = []
        for factor in range(1, factor_max + 1):
            decoded = self.decode_zoom(img, target, factor)
            data_multi.append(decoded[0])
            target_multi.append(decoded[1])

        return torch.cat(data_multi), torch.cat(target_multi)

    def decode_zoom_bound(self, img, target, factor_max, bound=128):
        """Uniform multi-formation with bounded number of synthetic data
        """
        bound_cur = bound - len(img)
        budget = len(img)

        data_multi = []
        target_multi = []

        idx = 0
        decoded_total = 0
        for factor in range(factor_max, 0, -1):
            decode_size = factor**2
            if factor > 1:
                n = min(bound_cur // decode_size, budget)
            else:
                n = budget

            decoded = self.decode_zoom(img[idx:idx + n], target[idx:idx + n], factor)
            data_multi.append(decoded[0])
            target_multi.append(decoded[1])

            idx += n
            budget -= n
            decoded_total += n * decode_size
            bound_cur = bound - decoded_total - budget

            if budget == 0:
                break

        data_multi = torch.cat(data_multi)
        target_multi = torch.cat(target_multi)
        return data_multi, target_multi

    def decode(self, data, target, bound=128):
        """Multi-formation
        """
        if self.factor > 1:
            if self.decode_type == 'multi':
                data, target = self.decode_zoom_multi(data, target, self.factor)
            elif self.decode_type == 'bound':
                data, target = self.decode_zoom_bound(data, target, self.factor, bound=bound)
            else:
                data, target = self.decode_zoom(data, target, self.factor)

        return data, target

    def sample(self, c, max_size=128):
        """Sample synthetic data per class
        """
        idx_from = self.ipc * c
        idx_to = self.ipc * (c + 1)
        data = self.data[idx_from:idx_to]
        target = self.targets[idx_from:idx_to]

        data, target = self.decode(data, target, bound=max_size)
        data, target = self.subsample(data, target, max_size=max_size)

        if self.soft_label is not None:
            target = [int(t) for t in target]
            target_soft = torch.tensor([self.soft_label[t].tolist() for t in target],
                                        requires_grad=False,
                                        device=self.device)
            
            return data, target, target_soft
            
        return data, target

    def loader(self, args, augment=True):
        """Data loader for condensed data
        """
        if args.dataset == 'imagenet':
            train_transform, _ = transform_imagenet(augment=augment,
                                                    from_tensor=True,
                                                    size=0,
                                                    rrc=args.rrc,
                                                    rrc_size=self.size[0])
        elif args.dataset[:5] == 'cifar':
            train_transform, _ = transform_cifar(augment=augment, from_tensor=True)
        elif args.dataset == 'svhn':
            train_transform, _ = transform_svhn(augment=augment, from_tensor=True)
        elif args.dataset == 'mnist':
            train_transform, _ = transform_mnist(augment=augment, from_tensor=True)
        elif args.dataset == 'fashion':
            train_transform, _ = transform_fashion(augment=augment, from_tensor=True)

        data_dec = []
        target_dec = []
        for c in range(self.nclass):
            idx_from = self.ipc * c
            idx_to = self.ipc * (c + 1)
            data = self.data[idx_from:idx_to].detach()
            target = self.targets[idx_from:idx_to].detach()
            data, target = self.decode(data, target)

            data_dec.append(data)
            target_dec.append(target)

        data_dec = torch.cat(data_dec)
        target_dec = torch.cat(target_dec)

        train_dataset = TensorDataset(data_dec.cpu(), target_dec.cpu(), train_transform)

        print("Decode condensed data: ", data_dec.shape)
        nw = 0 if not augment else args.workers
        train_loader = MultiEpochsDataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=nw,
                                             persistent_workers=nw > 0)
        return train_loader

    def test(self, args, val_loader, logger, bench=True):
        """Condensed data evaluation
        """
        loader = self.loader(args, args.augment)
        best_acc, _, _, _ = test_data(args, loader, val_loader, test_resnet=False, logger=logger)

        if bench and not (args.dataset in ['mnist', 'fashion']):
            resnet_acc, _, _, _ = test_data(args, loader, val_loader, test_resnet=True, logger=logger)
            
            return [best_acc, resnet_acc]
        
        return [best_acc]

class Soft_dataset(torch.utils.data.Dataset):
    def __init__(self, train_dataset, teacher, temperature = None):
        self.data = train_dataset
        self.teacher = teacher
        self.nclass = train_dataset.nclass
        self.targets = train_dataset.targets
        self.soft_targets = torch.empty((0, 100), dtype=torch.float32)
        self.temperature = temperature

        softmax = torch.nn.Softmax()
        for v in tqdm(self.data):
            pred = teacher(v[0].unsqueeze(0).to("cuda")).detach().to("cpu")
            if self.temperature:
                pred = pred/self.temperature
            self.soft_targets = torch.vstack((self.soft_targets, softmax(pred)))
            
            
    def __getitem__(self, index):
        img, target, soft_target = self.data[index][0], self.targets[index], self.soft_targets[index]
        return img, target, soft_target

    def __len__(self):
        return len(self.data)

def load_resized_data(args, soft_label = None):
    """Load original training data (fixed spatial size and without augmentation) for condensation
    """
    if soft_label:
        
        train_dataset = datasets.CIFAR100(args.data_dir,
                                          train=True,
                                          transform=transforms.ToTensor())

        normalize = transforms.Normalize(mean=MEANS['cifar100'], std=STDS['cifar100'])
        train_dataset.nclass = 100
        args.net_type = 'resnet'
        args.depth = 50

        # teacher = define_model(args, train_dataset.nclass, teacher = True).to("cuda")
        # teacher.load_state_dict(torch.load("pytorch-cifar100/checkpoint/resnet50/Tuesday_04_April_2023_11h_58m_31s/resnet50-193-best.pth"))
        teacher = define_models(args, train_dataset.nclass).to("cuda")
        teacher.load_state_dict(torch.load("results/cifar100/resnet50in_cut/model_best.pth.tar")['state_dict'])

        soft_dataset = Soft_dataset(train_dataset, teacher, args.temperature)

        return soft_dataset

    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(args.data_dir, train=True, transform=transforms.ToTensor())
        normalize = transforms.Normalize(mean=MEANS['cifar10'], std=STDS['cifar10'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        val_dataset = datasets.CIFAR10(args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 10

    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(args.data_dir,
                                          train=True,
                                          transform=transforms.ToTensor())

        normalize = transforms.Normalize(mean=MEANS['cifar100'], std=STDS['cifar100'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        val_dataset = datasets.CIFAR100(args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 100

    elif args.dataset == 'svhn':
        train_dataset = datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                      split='train',
                                      transform=transforms.ToTensor())
        train_dataset.targets = train_dataset.labels

        normalize = transforms.Normalize(mean=MEANS['svhn'], std=STDS['svhn'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        val_dataset = datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                    split='test',
                                    transform=transform_test)
        train_dataset.nclass = 10

    elif args.dataset == 'mnist':
        train_dataset = datasets.MNIST(args.data_dir, train=True, transform=transforms.ToTensor())

        normalize = transforms.Normalize(mean=MEANS['mnist'], std=STDS['mnist'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        val_dataset = datasets.MNIST(args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 10

    elif args.dataset == 'fashion':
        train_dataset = datasets.FashionMNIST(args.data_dir,
                                              train=True,
                                              transform=transforms.ToTensor())

        normalize = transforms.Normalize(mean=MEANS['fashion'], std=STDS['fashion'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        val_dataset = datasets.FashionMNIST(args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 10

    elif args.dataset == 'imagenet':
        traindir = os.path.join(args.imagenet_dir, 'train')
        valdir = os.path.join(args.imagenet_dir, 'val')

        # We preprocess images to the fixed size (default: 224)
        resize = transforms.Compose([
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.PILToTensor()
        ])

        if args.load_memory:  # uint8
            transform = None
            load_transform = resize
        else:
            transform = transforms.Compose([resize, transforms.ConvertImageDtype(torch.float)])
            load_transform = None

        _, test_transform = transform_imagenet(size=args.size)
        train_dataset = ImageFolder(traindir,
                                    transform=transform,
                                    nclass=args.nclass,
                                    phase=args.phase,
                                    seed=args.dseed,
                                    load_memory=args.load_memory,
                                    load_transform=load_transform)
        val_dataset = ImageFolder(valdir,
                                  test_transform,
                                  nclass=args.nclass,
                                  phase=args.phase,
                                  seed=args.dseed,
                                  load_memory=False)

    val_loader = MultiEpochsDataLoader(val_dataset,
                                       batch_size=args.batch_size // 2,
                                       shuffle=False,
                                       persistent_workers=True,
                                       num_workers=4)

    assert train_dataset[0][0].shape[-1] == val_dataset[0][0].shape[-1]  # width check

    return train_dataset, val_loader


def remove_aug(augtype, remove_aug):
    aug_list = []
    for aug in augtype.split("_"):
        if aug not in remove_aug.split("_"):
            aug_list.append(aug)

    return "_".join(aug_list)


def diffaug(args, device='cuda'):
    """Differentiable augmentation for condensation
    """
    aug_type = args.aug_type
    normalize = utils.Normalize(mean=MEANS[args.dataset], std=STDS[args.dataset], device=device)
    print("Augmentataion Matching: ", aug_type)
    augment = DiffAug(strategy=aug_type, batch=True)
    aug_batch = transforms.Compose([normalize, augment])

    if args.mixup_net == 'cut':
        aug_type = remove_aug(aug_type, 'cutout')
    print("Augmentataion Net update: ", aug_type)
    augment_rand = DiffAug(strategy=aug_type, batch=False)
    aug_rand = transforms.Compose([normalize, augment_rand])

    return aug_batch, aug_rand


def dist(x, y, method='mse'):
    """Distance objectives
    """
    if method == 'mse':
        dist_ = (x - y).pow(2).sum()
    elif method == 'l1':
        dist_ = (x - y).abs().sum()
    elif method == 'l1_mean':
        n_b = x.shape[0]
        dist_ = (x - y).abs().reshape(n_b, -1).mean(-1).sum()
    elif method == 'cos':
        x = x.reshape(x.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        
        dist_ = torch.mean(1 - torch.sum(x * y, dim=-1) /
                          (torch.norm(x, dim=-1) * torch.norm(y, dim=-1) + 1e-6)).item()

    return dist_


def add_loss(loss_sum, loss):
    if loss_sum == None:
        return loss
    else:
        return loss_sum + loss

def normalize(img):
    return (img - img.min())/(img.max()-img.min())

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1, weight = None):
        """if smoothing == 0, it's one-hot method
           if 0 < smoothing < 1, it's smooth method
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        assert 0 <= self.smoothing < 1
        pred = pred.log_softmax(dim=self.dim)

        if self.weight is not None:
            pred = pred * self.weight.unsqueeze(0)   

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

def matchloss(args, img_real, img_syn, lab_real, lab_syn, model):
    """Matching losses (feature or gradient)
    """
    loss = torch.tensor([0.0], requires_grad = True).cuda()
    cosSimi = []
    

    if args.match == 'feat':
        with torch.no_grad():
            feat_tg = model.get_feature(img_real, args.idx_from, args.idx_to)
        feat = model.get_feature(img_syn, args.idx_from, args.idx_to)

        for i in range(len(feat)):
            loss = add_loss(loss, dist(feat_tg[i].mean(0), feat[i].mean(0), method=args.metric))
    elif args.match == 'grad':
        criterion = nn.CrossEntropyLoss()

        output_real = model(img_real)
        loss_real = criterion(output_real, lab_real)
        g_real = torch.autograd.grad(loss_real, model.parameters())
        g_real = list((g.detach() for g in g_real))

        criterion_syn = nn.CrossEntropyLoss()
        output_syn = model(img_syn)
        loss_syn = criterion_syn(output_syn, lab_syn)
        g_syn = torch.autograd.grad(loss_syn, model.parameters(), create_graph=True)

        if args.capacity_thres >0:
            for i in range(len(g_real)):
                if (len(g_real[i].shape) == 1) and not args.bias:  # bias, normliazation
                    continue
                if (len(g_real[i].shape) == 2) and not args.fc:
                    continue

                similarity = 1 - dist(g_real[i], g_syn[i], method="cos")
                if similarity > args.capacity_thres:
                    loss = add_loss(loss, dist(g_real[i], g_syn[i], method=args.metric))
                cosSimi.append(similarity)
            return loss, sum(cosSimi)/len(cosSimi)
        else:
            for i in range(len(g_real)):
                if (len(g_real[i].shape) == 1) and not args.bias:  # bias, normliazation
                    continue
                if (len(g_real[i].shape) == 2) and not args.fc:
                    continue

                loss = add_loss(loss, dist(g_real[i], g_syn[i], method=args.metric))


            return loss, g_real

    elif args.match == 'sp_grad':
        criterion = nn.CrossEntropyLoss()

        output_real = model(img_real)
        loss_real = criterion(output_real, lab_real)
        g_real = torch.autograd.grad(loss_real, model.parameters())
        g_real = list((g.detach() for g in g_real))
        # g_img_real = torch.autograd.grad(loss_real, img_real)[0]
        # expert_saliencymaps, _ = torch.max(g_img_real.abs(), dim = 1)

        criterion_syn = nn.CrossEntropyLoss()
        output_syn = model(img_syn)
        loss_syn = criterion_syn(output_syn, lab_syn)
        g_syn = torch.autograd.grad(loss_syn, model.parameters(), create_graph=True, retain_graph = True)
        g_syn_img = torch.autograd.grad(loss_syn, img_syn, retain_graph = True)[0]

        student_saliencymaps, _ = torch.max(g_syn_img.abs(), dim = 1)


        for i in range(len(g_real)):
            if (len(g_real[i].shape) == 1) and not args.bias:  # bias, normliazation
                continue
            if (len(g_real[i].shape) == 2) and not args.fc:
                continue

            loss = add_loss(loss, dist(g_real[i], g_syn[i], method=args.metric))

        return loss, student_saliencymaps.squeeze()

    elif args.match == 'soft_grad_teacher':
    
        criterion = nn.CrossEntropyLoss()

        output_real = model(img_real)
        loss_real = criterion(output_real, lab_real)
        g_real = torch.autograd.grad(loss_real, model.parameters())
        g_real = list((g.detach() for g in g_real))

        criterion_syn = nn.CrossEntropyLoss()
        output_syn = model(img_syn)
        loss_syn = criterion_syn(output_syn, lab_syn)
        g_syn = torch.autograd.grad(loss_syn, model.parameters(), create_graph=True)

        for i in range(len(g_real)):
            if (len(g_real[i].shape) == 1) and not args.bias:  # bias, normliazation
                continue
            if (len(g_real[i].shape) == 2) and not args.fc:
                continue

            loss = add_loss(loss, dist(g_real[i], g_syn[i], method=args.metric))

        return loss
    elif args.match == 'soft_grad_noise':
    
        criterion = LabelSmoothingLoss(classes = args.nclass, smoothing = args.smoothing)

        output_real = model(img_real)
        loss_real = criterion(output_real, lab_real)
        g_real = torch.autograd.grad(loss_real, model.parameters())
        g_real = list((g.detach() for g in g_real))

        if args.syn_soft_label:
            criterion_syn = LabelSmoothingLoss(classes = args.nclass, smoothing = args.smoothing)
        else:
            criterion_syn = LabelSmoothingLoss(classes = args.nclass, smoothing = 0)
            
        output_syn = model(img_syn)
        loss_syn = criterion_syn(output_syn, lab_syn)
        g_syn = torch.autograd.grad(loss_syn, model.parameters(), create_graph=True)

        for i in range(len(g_real)):
            if (len(g_real[i].shape) == 1) and not args.bias:  # bias, normliazation
                continue
            if (len(g_real[i].shape) == 2) and not args.fc:
                continue

            loss = add_loss(loss, dist(g_real[i], g_syn[i], method=args.metric))

        return loss, g_real
    elif args.match == 'soft_label':
        criterion = nn.CrossEntropyLoss(label_smoothing = args.smoothing)

        output_real = model(img_real)
        loss_real = criterion(output_real, lab_real)
        g_real = torch.autograd.grad(loss_real, model.parameters())
        g_real = list((g.detach() for g in g_real))

        criterion_syn = nn.CrossEntropyLoss(label_smoothing = args.smoothing)
        output_syn = model(img_syn)
        loss_syn = criterion_syn(output_syn, lab_syn)
        g_syn = torch.autograd.grad(loss_syn, model.parameters(), create_graph=True)

        for i in range(len(g_real)):
            if (len(g_real[i].shape) == 1) and not args.bias:  # bias, normliazation
                continue
            if (len(g_real[i].shape) == 2) and not args.fc:
                continue

            loss = add_loss(loss, dist(g_real[i], g_syn[i], method=args.metric))


        return loss, g_real


# pretrained model sampled by epoch or acc range
def pretrain_sample(args, model, verbose=False):
    """Load pretrained networks
    """
    if args.sample_accrange:
        folder_base = f'./pretrained/{args.datatag}/{args.modeltag}_cut_pre/'
        folder_list = glob.glob(f'{folder_base}*{args.sample_accrange[0]}_{args.sample_accrange[1]}')
        tag = np.random.randint(len(folder_list))
        folder = folder_list[tag]

        file_list = glob.glob(f'{folder}/checkpoint*.pth.tar')
        tag_file = np.random.randint(len(file_list))
        ckpt = file_list[tag_file]

        file_dir = ckpt
        print(f"Expert: pretrain from : {file_dir}")
        load_ckpt(model, file_dir , verbose=verbose)


    else:
        folder_base = f'./pretrained/{args.datatag}/{args.modeltag}_cut_pre'
        folder_list = glob.glob(f'{folder_base}*')
        tag = np.random.randint(len(folder_list))
        folder = folder_list[tag]

        epoch = args.pt_from
        if args.pt_num > 1:
            epoch = np.random.randint(args.pt_from, args.pt_from + args.pt_num)
        ckpt = f'checkpoint{epoch}.pth.tar'

        
        file_dir = os.path.join(folder, ckpt)
        print(f"Expert: pretrain from : {file_dir}")
        load_ckpt(model, file_dir, verbose=verbose)

def add_dropout(model):
    print(f"convnet_dropout, ratio:{args.dropout_rate}, dataset:{args.dataset}, depth = {args.depth}, norm_type: {args.norm_type}, nch = {args.nch}")
    pretrained_dict = model.state_dict()

    width = int(128 * args.width)
    model_dropout = CN.ConvNet_dropout(args.nclass,
                    net_norm=args.norm_type,
                    net_depth=args.depth,
                    net_width=width,
                    channel=args.nch,
                    dropout_rate = args.dropout_rate,
                    im_size=(args.size, args.size))

    model_dropout_dict = model_dropout.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dropout_dict}
    model_dropout_dict.update(pretrained_dict)

    model_dropout.load_state_dict(model_dropout_dict)
    return model_dropout
def apply_dropout(args, model, device):
    pretrained_dict = model.state_dict()
    width = int(128 * args.width)
    model_dropout = CN.ConvNet_mask(args.nclass,
                    net_norm=args.norm_type,
                    net_depth=args.depth,
                    net_width=width,
                    channel=args.nch,
                    dropout_rate = args.dropout_rate,
                    im_size=(args.size, args.size), 
                    )

    model_dropout_dict = model_dropout.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dropout_dict}
    model_dropout_dict.update(pretrained_dict)

    model_dropout.load_state_dict(model_dropout_dict)
    return model_dropout


def apply_pruning(args, model):
    print(f"Pruning_type:{args.pruning_type}, strategy:{args.pruning_ratio_type}")
    if args.pruning_type == "fc":
        parameters_to_prune = [
        (model.classifier, 'weight')
        ]
    elif args.pruning_type == "global":
        parameters_to_prune = [
            (model.layers.conv[0], 'weight'),
            (model.layers.conv[1], 'weight'),
            (model.layers.conv[2], 'weight'),
            (model.classifier, 'weight'),
        ]
    # elif args.pruning_type == "pruning_shallow":
    #     parameters_to_prune = [
    #         (model.layers.conv[0], 'weight'),
    #         (model.layers.conv[1], 'weight'),
    #     ]
    # elif args.pruning_type == "pruning_deep":
    #     parameters_to_prune = [
    #         (model.layers.conv[2], 'weight'),
    #         (model.classifier, 'weight'),
    #     ]

    # if args.pruning_ratio_type == "uniform":
    for module, name in parameters_to_prune:
        if name == 'weight':
            prune.random_unstructured(module, name='weight', amount=args.pruning_ratio)
        else:
            prune.random_unstructured(module, name='bias', amount=args.pruning_ratio)
    # else:
    #     for idx, (module, name) in enumerate(parameters_to_prune):
    #         if idx<2:
    #             if name == 'weight':
    #                 prune.random_unstructured(module, name='weight', amount=args.pruning_ratio_shallow)
    #             else:
    #                 prune.random_unstructured(module, name='bias', amount=args.pruning_ratio_shallow)
    #         else:
    #             if name == 'weight':
    #                 prune.random_unstructured(module, name='weight', amount=args.pruning_ratio_deep)
    #             else:
    #                 prune.random_unstructured(module, name='bias', amount=args.pruning_ratio_deep)

def apply_masking(args, model):
    if args.masking_type == "fc":
        parameters_to_prune = [
        (model.classifier, 'weight')
        ]
    elif args.masking_type == "global":
        parameters_to_prune = [
            (model.layers.conv[0], 'weight'),
            (model.layers.conv[1], 'weight'),
            (model.layers.conv[2], 'weight'),
            (model.classifier, 'weight'),
        ]

    for module, name in parameters_to_prune:
        if name == 'weight':
            weight = module.weight
            weight_shape = weight.shape
            mask = torch.zeros(weight_shape)
            for i in range(weight_shape[0]):
                for j in range(weight_shape[1]):
                    if random.random() < args.pruning_ratio:
                        mask[i, j] = 0  # 刪除連結
                    else:
                        mask[i, j] = 1  # 保留連結
            mask = mask.to(weight.device)
            module.weight = nn.Parameter(weight * mask)
        else:
            bias = module.bias
            bias_shape = bias.shape
            mask = torch.zeros(bias_shape)
            for i in range(bias_shape[0]):
                if random.random() < args.pruning_ratio:
                    mask[i] = 0  # 删掉
                else:
                    mask[i] = 1  # 俩留
            mask = mask.to(bias.device)
            module.bias = nn.Parameter(bias * mask)
            


def condense(args, logger, device='cuda'):
    """Optimize condensed data
    """
    tensor2img = transforms.ToPILImage()
    # Define real dataset and loader
    trainset, val_loader = load_resized_data(args)
    if args.load_memory:
        loader_real = ClassMemDataLoader(trainset, batch_size=args.batch_real)
    else:
        loader_real = ClassDataLoader(trainset,
                                      batch_size=args.batch_real,
                                      num_workers=args.workers,
                                      shuffle=True,
                                      pin_memory=True,
                                      drop_last=True)

    args.tmp_net_type = args.net_type
    args.tmp_depth = args.depth

    
    if args.teacher_soft_label:
        trainset_soft = load_resized_data(args, args.teacher_soft_label)
        if args.load_memory:
            loader_real_st = ClassMemDataLoader(trainset_soft, batch_size=args.batch_real)
        else:
            loader_real_st = ClassDataLoader(trainset_soft,
                                        batc_size=args.batch_real,
                                        num_workers=args.workers,
                                        shuffle=True,
                                        pin_memory=True,
                                        drop_last=True) 
        if args.syn_soft_label:
            syn_soft_target = torch.empty((0, trainset_soft.nclass), dtype=torch.float32)
            for class_idx in range(trainset_soft.nclass):
                label_class = loader_real_st.class_sample(class_idx)[1]

                class_dis = torch.mean(label_class, axis = 0)
                class_dis = class_dis.cpu()
                syn_soft_target = torch.vstack((syn_soft_target, class_dis))
    
    args.net_type = args.tmp_net_type
    args.depth = args.tmp_depth

    nclass = trainset.nclass
    nch, hs, ws = trainset[0][0].shape

    # Define syn dataset
    if args.match == "soft_grad_teacher" and args.syn_soft_label:
        synset = Synthesizer(args, nclass, nch, hs, ws, soft_label = syn_soft_target)
    else:
        synset = Synthesizer(args, nclass, nch, hs, ws)

    synset.init(loader_real, init_type=args.init)
    save_img(os.path.join(args.save_dir, 'init.png'),
             synset.data,
             unnormalize=False,
             dataname=args.dataset)

    # Define augmentation function
    aug, aug_rand = diffaug(args)
    save_img(os.path.join(args.save_dir, f'aug.png'),
             aug(synset.sample(0, max_size=args.batch_syn_max)[0]),
             unnormalize=True,
             dataname=args.dataset)

    # if not args.test:
        # synset.test(args, val_loader, logger, bench=False)

    # Data distillation
    optim_img = torch.optim.SGD(synset.parameters(), lr=args.lr_img, momentum=args.mom_img, weight_decay = args.wd_img)

    ts = utils.TimeStamp(args.time)
    n_iter = args.niter * 100 // args.inner_loop
    it_log = n_iter // 50
    it_test = [n_iter//20, n_iter // 8, n_iter // 4, n_iter // 2, n_iter//4+n_iter//2 , n_iter]
    logger(f"\nevaluation for {it_test} iteration")
    
    logger(f"\nStart condensing with {args.match} matching for {n_iter} iteration")
    args.fix_iter = max(1, args.fix_iter)

    class2saliency = random.sample(range(100), 10)

    
    args.pretrained_dir = f'./pretrained/{args.datatag}/{args.modeltag}_cut'
    model_pool = []
    seeds = []
    if args.sample_accrange:
        path_model_pool = os.path.join(args.pretrained_dir, f"{args.pool_number}")
        path_model_pool = os.path.join(path_model_pool, f"{args.sample_accrange[0]}_{args.sample_accrange[1]}")
        model_pool = glob.glob(os.path.join(path_model_pool, "*.pth.tar"))
    
        logger(f"pretrained dir: {path_model_pool}")
    logger(f"model pool: {model_pool}")

    ## get all possible seeds _300_{seeds}_30_40
    # folder_name = os.path.join(args.pretrained_dir, f"*_40_50")
    # folders = glob.glob(folder_name)
    # for folder in folders:
    #     seeds.append(folder.split("/")[-1].split("_")[2])
    # random.shuffle(seeds)
    # logger(f"Seeds: {seeds}")

    # ## generate model pool: according to args.pool_number and args.sample_accrange,
    # ## random pick specific number of model from folder pretrained/{args.datatag}/{args.modeltag}_cut/_300_{random_seed}_{args.sample_accrange[0]}_{args.sample_accrange[1]}/
    # if args.pool_number and args.pool_number > 0:
    #     if args.sample_accrange[1] - args.sample_accrange[0] == 0:
    #         for seed in seeds:
    #             folder_name = os.path.join(args.pretrained_dir, f"*_{seed}_initial")
    #             folder = glob.glob(folder_name)[0]
    #             logger(f"folder : {folder}")
    #             model_pool.append(random.sample(glob.glob(os.path.join(folder, "*.pth.tar")), 1)[0])
    #     elif len(args.sample_accrange) == 2 and args.distributed_num == None:
    #         number_of_range = (args.sample_accrange[1] - args.sample_accrange[0])//10
        
    #         ## use number_of_range(3) to evenly distribute pool_number(10), e.g [3,4,3]
    #         num_of_each_range = [args.pool_number//number_of_range for _ in range(number_of_range)] 
    #         for i in range(args.pool_number%number_of_range):
    #             num_of_each_range[i] += 1
    #         random.shuffle(num_of_each_range)
    #         logger(f"num_of_each_range: {num_of_each_range}")
            
    #         dis_seeds = []
    #         for num in num_of_each_range:
    #             dis_seeds.append(seeds[:num])
    #             seeds = seeds[num:]

    #         for ith, seed_in_folders in enumerate(dis_seeds):
    #             for seed in seed_in_folders:
    #                 folder_name = os.path.join(args.pretrained_dir, f"*_{seed}_{args.sample_accrange[0]+ith*10}_{args.sample_accrange[0]+ith*10+10}")

    #                 ## get folders by folder_name and sample args.pool_number of folders randomly
    #                 folder = glob.glob(folder_name)[0]
    #                 logger(f"folder : {folder}")
    #                 model_pool.append(random.sample(glob.glob(os.path.join(folder, "*.pth.tar")), 1)[0])
    #     elif args.distributed_num:
    #         number_of_range = len(args.sample_accrange)//2
    #         num_of_each_range = args.distributed_num
    #         dis_seeds = []
    #         for num in num_of_each_range:
    #             dis_seeds.append(seeds[:num])
    #             seeds = seeds[num:]

    #         for ith, seed_in_folders in enumerate(dis_seeds):
    #             for seed in seed_in_folders:
    #                 folder_name = os.path.join(args.pretrained_dir, f"*_{seed}_{args.sample_accrange[0 + ith*2]}_{args.sample_accrange[1+ ith*2]}")

    #                 ## get folders by folder_name and sample args.pool_number of folders randomly
    #                 folder = glob.glob(folder_name)[0]
    #                 logger(f"folder : {folder}")
    #                 model_pool.append(random.sample(glob.glob(os.path.join(folder, "*.pth.tar")), 1)[0])

    #     else:
    #         number_of_range = len(args.sample_accrange)//2

    #         num_of_each_range = [args.pool_number//number_of_range for _ in range(number_of_range)] 
    #         for i in range(args.pool_number%number_of_range):
    #             num_of_each_range[i] += 1
    #         random.shuffle(num_of_each_range)
    #         logger(f"num_of_each_range: {num_of_each_range}")
            
    #         dis_seeds = []
    #         for num in num_of_each_range:
    #             dis_seeds.append(seeds[:num])
    #             seeds = seeds[num:]

    #         for ith, seed_in_folders in enumerate(dis_seeds):
    #             for seed in seed_in_folders:
    #                 folder_name = os.path.join(args.pretrained_dir, f"*_{seed}_{args.sample_accrange[0 + ith*2]}_{args.sample_accrange[1+ ith*2]}")

    #                 ## get folders by folder_name and sample args.pool_number of folders randomly
    #                 folder = glob.glob(folder_name)[0]
    #                 logger(f"folder : {folder}")
    #                 model_pool.append(random.sample(glob.glob(os.path.join(folder, "*.pth.tar")), 1)[0])




        # logger(f"model pool: {model_pool}")



    for it in range(n_iter):
        wandb.log({"Progress": it, "ot_epoch":it})
        if it % args.fix_iter == 0:            
            model = define_model(args, nclass)
            if args.dist:
                model = torch.nn.DataParallel(model).to(device)
            else:
                model = model.to(device)

            model.train()
            optim_net = optim.SGD(model.parameters(),
                                  args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
            criterion = nn.CrossEntropyLoss()

            if len(model_pool) > 0:
                ## random pick one model from model_pool and load it
                model_path = random.sample(model_pool, 1)[0]
                load_ckpt(model, model_path, verbose=False)

                logger(f"load model from {model_path}")

            elif args.pt_from >= 0 or args.sample_accrange:
                pretrain_sample(args, model)
            if args.early > 0:
                print(f"Expert: Early epoch:{args.early}")
                for _ in range(args.early):
                    train_epoch(args,
                                loader_real,
                                model,
                                criterion,
                                optim_net,
                                aug=aug_rand,
                                mixup=args.mixup_net)
            

        loss_total = 0
        image_sps = []
        images = []

        synset.data.data = torch.clamp(synset.data.data, min=0., max=1.)


        # for calculate gradient bwtween innerloop
        inner_loop_simi = [] # (100, 100) (innerloop, classes_num)
        last_classes_grad = None
        
        # ============== model augmentation =================
        if args.apply_pruning:
            logger(f'Pruning Type: {args.pruning_type}, ratio: {args.pruning_ratio}')
            apply_pruning(args, model)
        elif args.apply_dropout:
            # model = add_dropout(model).to(device)
            model = apply_dropout(args, model, device).to(device)
        elif args.apply_masking:
            apply_masking(args, model)

        
        if args.num_tune_subnetwork!=0:
            for _ in range(args.num_tune_subnetwork):
                    train_epoch(args,
                                loader_real,
                                model,
                                criterion,
                                optim_net,
                                aug=aug_rand,
                                mixup=args.mixup_net)

        # ===================================================
        
        
        for ot in tqdm(range(args.inner_loop)):
            ts.set()
            
            if args.inner_schedule and ot in inner_schedule:
                # train one epoch
                if args.n_data > 0:
                    for _ in range(args.net_epoch):
                        top1, top5, losses = train_epoch(args,
                                                        loader_real,
                                                        model,
                                                        criterion,
                                                        optim_net,
                                                        n_data=args.n_data,
                                                        aug=aug_rand,
                                                        mixup=args.mixup_net)
                continue


            # (100, len(Weight))
            classes_grad = []
            inner_loss = 0
            classes_similarity = []

            for c in range(nclass):
                if args.teacher_soft_label:
                    img, lab = loader_real_st.class_sample(c)
                else:
                    img, lab = loader_real.class_sample(c)

                if args.match == "soft_grad_teacher" and  args.syn_soft_label:
                    img_syn, _, lab_syn = synset.sample(c, max_size=args.batch_syn_max)
                else:
                    img_syn, lab_syn = synset.sample(c, max_size=args.batch_syn_max)
                    
                ts.stamp("data")

                if args.saliency and (it+1) in it_test and ot==(args.inner_loop-1):
                    img_syn.retain_grad()

                n = img.shape[0]
                img_aug = aug(torch.cat([img, img_syn]))
                ts.stamp("aug")

                if args.capacity_thres > 0:
                    loss, similarity = matchloss(args, img_aug[:n], img_aug[n:], lab, lab_syn, model)
                    classes_similarity.append(similarity)
                    
                else:    
                    loss, grad = matchloss(args, img_aug[:n], img_aug[n:], lab, lab_syn, model)

               

                inner_loss += loss.item()
                loss_total += loss.item()
                ts.stamp("loss")

                optim_img.zero_grad()
                loss.backward()

                # SaliencyMap for syn image update
                if args.saliency and (it+1) in it_test and ot==(args.inner_loop-1):
                    g_img_syn = img_syn.grad
                    syn_saliency = torch.max(g_img_syn.abs(), dim = 1)[0].squeeze()


                
                optim_img.step()
                ts.stamp("backward")

                
                #saliencymap
                if args.saliency and (it+1) in it_test and ot==(args.inner_loop-1): ##
            
                    syn_saliency = syn_saliency.cpu().numpy()
                    my_cmap = matplotlib.cm.get_cmap('jet')

                    i = 1
                    for imgs, ss in zip(img_syn.detach(), syn_saliency):
                        n_saliency = (ss - np.min(ss))/ (np.max(ss) - np.min(ss))
                        color_array = my_cmap(n_saliency)
                        img_de = torch.clamp(imgs.unsqueeze(0), min=0., max=1.).cpu()
                        image_sp = wandb.Image(color_array, caption = f"img{c}_{i}_{it}")
                        image = wandb.Image(img_de, caption = f"img{c}_{i}_{it}")
                        image_sps.append(image_sp)
                        images.append(image)
                        i+=1
                        
            
            # wandb.log({ "inner_epoch": ot+it*args.inner_loop})
            if args.grad_simi:
                if last_classes_grad is not None:
                    classes_simi = []
                    for class_idx in range(len(classes_grad)):
                        g_real_class = classes_grad[class_idx]
                        last_grad_class = last_classes_grad[class_idx]
                        similarity = []

                        for i in range(len(g_real_class)):
                            if (len(g_real_class[i].shape) == 1) and not args.bias:  # bias, normliazation
                                continue
                            if (len(g_real_class[i].shape) == 2) and not args.fc:
                                continue

                            simi = dist(g_real_class[i], last_grad_class[i], method = "cos")
                            similarity.append(simi)
                        classes_simi.append(np.mean(similarity))

                    # inner_loop_simi : (99, 100) (innerloop-1, classes_num)    
                    inner_loop_simi.append(classes_simi)
                    last_classes_grad = classes_grad
                else:
                    last_classes_grad = classes_grad

                

            # Net update
            if args.n_data > 0:
                for _ in range(args.net_epoch):
                    top1, top5, losses = train_epoch(args,
                                                    loader_real,
                                                    model,
                                                    criterion,
                                                    optim_net,
                                                    n_data=args.n_data,
                                                    aug=aug_rand,
                                                    mixup=args.mixup_net)
                    
                    if args.capacity_thres>0:
                        wandb.log({"InnerLoop/loss":inner_loss,
                                "Expert/inner/top1":top1,
                                "Expert/inner/top5":top5,
                                "Expert/inner/loss":losses,
                                "Similarity": sum(classes_similarity)/len(classes_similarity),
                                "inner_epoch": ot+it*args.inner_loop
                                })
                    else:
                            wandb.log({"InnerLoop/loss":inner_loss,
                                "Expert/inner/top1":top1,
                                "Expert/inner/top5":top5,
                                "Expert/inner/loss":losses,
                                "inner_epoch": ot+it*args.inner_loop
                                })

           
            
        

        
            ts.stamp("net update")

            if (ot + 1) % 10 == 0:
                ts.flush()

        # Logging
        if args.n_data>0:
            wandb.log({"Expert/top1":top1, 
                        "Expert/top5":top5, 
                        "Expert/loss":losses, "ot_epoch":it})

        if args.grad_simi:
            print(f"save boxplot: ({len(inner_loop_simi)}, {len(inner_loop_simi[0])})")
            # plot inner_loop_simi boxplot:
            plt.figure(figsize=(30,15))
            plt.boxplot(inner_loop_simi)
            plt.show()
            plt.savefig(f"{args.save_dir}/{it}_inner_loop_simi.png")



        wandb.log({"MatchLoss":loss_total/nclass/args.inner_loop, "ot_epoch":it})
        
        
                
        if it % it_log == 0:
            logger(
                f"{utils.get_time()} (Iter {it:3d}) loss: {loss_total/nclass/args.inner_loop:.1f}")
        if (it + 1) in it_test:
            save_img(os.path.join(args.save_dir, f'img{it+1}.png'),
                     synset.data,
                     unnormalize=False,
                     dataname=args.dataset)

            if args.saliency:
                for img, img_sp in tzip(images, image_sps):
                    wandb.log({"Student/Saliency": img_sp, "Student/image": img})
                

            # It is okay to clamp data to [0, 1] at here.
            # synset.data.data = torch.clamp(synset.data.data, min=0., max=1.)
            torch.save(
                [synset.data.detach().cpu(), synset.targets.cpu()],
                os.path.join(args.save_dir, f'data{it+1}.pt'))
            print("img and data saved!")
            
            if not args.test:
                test_result = synset.test(args, val_loader, logger)

                if len(test_result) >1:
                    wandb.log({"Test/ConvNet":test_result[0], 
                               "Test/ResNet":test_result[1], "ot_epoch":it})
                else:
                    wandb.log({"Test/ConvNet":test_result[0], "ot_epoch":it})

if __name__ == '__main__':
    import shutil
    from misc.utils import Logger
    from argument import args
    import torch.backends.cudnn as cudnn
    import json
    import wandb

    assert args.ipc > 0

    # if args.sample_accrange:
    #     if len(args.sample_accrange) ==2 :
    #         wandb.init(sync_tensorboard=False,
    #             project = "DatasetDistillation",
    #             job_type = "CleanRepo",
    #             config = args,
    #             name = f'IDC_dataset:{args.dataset}_ipc_ipc:{args.ipc}_model:{args.modeltag}_match:{args.match}_strategy:{args.strategy}_softlabel:{args.smoothing}_temperature:{args.temperature}_early:{args.early}_ptrange:({args.pt_from} - {args.pt_from+args.pt_num})_sampleRange:({args.sample_accrange[0]} - {args.sample_accrange[1]})_poolnumber:{args.pool_number}_dropout:{args.apply_dropout}_rate:{args.dropout_rate}_pruning:{args.apply_pruning}({args.pruning_ratio})_pruning_type:{args.pruning_type}_tuneEpoch:{args.num_tune_subnetwork}_Masking:{args.apply_masking}_masking_type:{args.masking_type}_capacity_thresh:{args.capacity_thres}')
    #     elif len(args.sample_accrange) ==4:
    #         if args.distributed_num:
    #             wandb.init(sync_tensorboard=False,
    #             project = "DatasetDistillation",
    #             job_type = "CleanRepo",
    #             config = args,
    #             name = f'IDC_dataset:{args.dataset}_ipc_ipc:{args.ipc}_model:{args.modeltag}_match:{args.match}_strategy:{args.strategy}_softlabel:{args.smoothing}_temperature:{args.temperature}_early:{args.early}_ptrange:({args.pt_from} - {args.pt_from+args.pt_num})_sampleRange:({args.sample_accrange[0]} - {args.sample_accrange[1]} - {args.sample_accrange[2]} - {args.sample_accrange[3]})_poolnumber:{args.pool_number}_dis_num:{args.distributed_num}_dropout:{args.apply_dropout}_rate:{args.dropout_rate}_pruning:{args.apply_pruning}({args.pruning_ratio})_pruning_type:{args.pruning_type}_Masking:{args.apply_masking}_masking_type:{args.masking_type}')
    #         else:
    #             wandb.init(sync_tensorboard=False,
    #             project = "DatasetDistillation",
    #             job_type = "CleanRepo",
    #             config = args,
    #             name = f'IDC_dataset:{args.dataset}_ipc_ipc:{args.ipc}_model:{args.modeltag}_match:{args.match}_strategy:{args.strategy}_softlabel:{args.smoothing}_temperature:{args.temperature}_early:{args.early}_ptrange:({args.pt_from} - {args.pt_from+args.pt_num})_sampleRange:({args.sample_accrange[0]} - {args.sample_accrange[1]} - {args.sample_accrange[2]} - {args.sample_accrange[3]})_poolnumber:{args.pool_number}_dropout:{args.apply_dropout}_rate:{args.dropout_rate}_pruning:{args.apply_pruning}({args.pruning_ratio})_pruning_type:{args.pruning_type}_Masking:{args.apply_masking}_masking_type:{args.masking_type}')
            


    # else:
    #      wandb.init(sync_tensorboard=False,
    #             project = "DatasetDistillation",
    #             job_type = "CleanRepo",
    #             config = args,
    #             name = f'IDC_dataset:{args.dataset}_ipc_ipc:{args.ipc}_model:{args.modeltag}_match:{args.match}_strategy:{args.strategy}_softlabel:{args.smoothing}_temperature:{args.temperature}_early:{args.early}_ptrange:({args.pt_from} - {args.pt_from+args.pt_num})_sampleRange:()_poolnumber:{args.pool_number}_dropout:{args.apply_dropout}_rate:{args.dropout_rate}_pruning:{args.apply_pruning}({args.pruning_ratio})_pruning_type:{args.pruning_type}_Masking:{args.apply_masking}_masking_type:{args.masking_type}')

    
    wandb.init(sync_tensorboard=False,
                project = "DatasetDistillation",
                job_type = "CleanRepo",
                config = args,
                name = f'IDC_dataset_{args.dataset}{args.tag}')

    cudnn.benchmark = True
    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    cur_file = os.path.join(os.getcwd(), __file__)
    shutil.copy(cur_file, args.save_dir)

    logger = Logger(args.save_dir)
    logger(f"Save dir: {args.save_dir}")
    with open(os.path.join(args.save_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    condense(args, logger)