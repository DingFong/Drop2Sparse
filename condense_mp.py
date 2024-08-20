import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from data import transform_imagenet, transform_cifar, TensorDataset, ImageFolder, save_img
from data import ClassPartMemDataLoader, MultiEpochsDataLoader
from data import MEANS, STDS
from train import define_model, train_epoch
from test import test_data, load_ckpt
from misc import utils
from misc.augment import DiffAug
from math import ceil
import glob

from tqdm import tqdm
from tqdm.contrib import tzip
import matplotlib.pyplot as plt
import matplotlib
import random
import torch.nn.utils.prune as prune
import shutil

class Synthesizer():
    """Condensed data class
    """
    def __init__(self, args, nclass_sub, subclass_list, nchannel, hs, ws, device='cuda'):
        self.ipc = args.ipc
        self.nclass_sub = nclass_sub
        self.subclass_list = subclass_list

        self.nchannel = nchannel
        self.size = (hs, ws)
        self.device = device

        self.data = torch.randn(size=(self.nclass_sub * self.ipc, self.nchannel, hs, ws),
                                dtype=torch.float,
                                requires_grad=True,
                                device=self.device)
        self.data.data = torch.clamp(self.data.data / 4 + 0.5, min=0., max=1.)
        self.targets = torch.tensor(
            [np.ones(self.ipc) * self.subclass_list[c] for c in range(nclass_sub)],
            dtype=torch.long,
            requires_grad=False,
            device=self.device).view(-1)
        print("\nDefine synthetic data: ", self.data.shape)

        self.factor = max(1, args.factor)
        self.decode_type = args.decode_type
        self.resize = nn.Upsample(size=self.size, mode='bilinear')
        print("Factor: ", self.factor)

    def init(self, loader, init_type='noise'):
        """Condensed data initialization
        """
        if init_type == 'random':
            print("Random initialize synset")
            for c in range(self.nclass_sub):
                img, _ = loader.class_sample(self.subclass_list[c], self.ipc)
                self.data.data[self.ipc * c:self.ipc * (c + 1)] = img.data.to(self.device)

        elif init_type == 'mix':
            print("Mixed initialize synset")
            for c in range(self.nclass_sub):
                img, _ = loader.class_sample(self.subclass_list[c], self.ipc * self.factor**2)
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

        data, target = self.decode(self.data.detach(), self.targets.detach())
        train_dataset = TensorDataset(data.cpu(), target.cpu(), train_transform)

        print("Decode condensed data: ", data.shape)
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
        # Test on current model
        best_acc, _, _, _ = test_data(args, loader, val_loader, test_resnet=False, logger=logger)

        if bench:
            # Test on resnet-10 models
            resnet_acc, _, _, _ = test_data(args, loader, val_loader, test_resnet=True, logger=logger)
            return [best_acc, resnet_acc]
        
        return [best_acc]

def load_resized_data(args, subclass_list):
    """Load original training data (without augmentation) for condensation
    """
    if args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(args.data_dir,
                                          train=True,
                                          transform=transforms.ToTensor())
        normalize = transforms.Normalize(mean=MEANS['cifar100'], std=STDS['cifar100'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        val_dataset = datasets.CIFAR100(args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 100

    elif args.dataset == 'imagenet':
        traindir = os.path.join(args.imagenet_dir, 'train')
        valdir = os.path.join(args.imagenet_dir, 'val')

        resize = transforms.Compose([
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.PILToTensor()
        ])

        if args.load_memory:  # uint8
            transform = None
            load_transform = resize
        else:
            transform = resize
            load_transform = None

        _, test_transform = transform_imagenet(size=args.size)
        train_dataset = ImageFolder(traindir,
                                    transform=transform,
                                    nclass=args.nclass,
                                    seed=args.dseed,
                                    load_memory=args.load_memory,
                                    load_transform=load_transform)
        val_dataset = ImageFolder(valdir,
                                  test_transform,
                                  nclass=args.nclass,
                                  seed=args.dseed,
                                  load_memory=False)

    # Validate only for the given subclass
    indices = []
    for i, c in enumerate(val_dataset.targets):
        if c in subclass_list:
            indices.append(i)

    val_dataset = torch.utils.data.Subset(val_dataset, indices)
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
        dist_ = torch.sum(1 - torch.sum(x * y, dim=-1) /
                          (torch.norm(x, dim=-1) * torch.norm(y, dim=-1) + 1e-6))

    return dist_


def add_loss(loss_sum, loss):
    if loss_sum == None:
        return loss
    else:
        return loss_sum + loss


def matchloss(args, img_real, img_syn, lab_real, lab_syn, model):
    """Matching losses (feature or gradient)
    """
    loss = None

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

        output_syn = model(img_syn)
        loss_syn = criterion(output_syn, lab_syn)
        g_syn = torch.autograd.grad(loss_syn, model.parameters(), create_graph=True)

        # save average gradient of each layer gradient
        layer_grad_real = []
        layer_grad_syn = []

        # save average gradient of each layer gradient
        layer_id = 0
        for i in range(len(g_real)):
            if (len(g_real[i].shape) == 1) and not args.bias:  # bias, normliazation
                continue
            if (len(g_real[i].shape) == 2) and not args.fc:
                continue

            loss = add_loss(loss, dist(g_real[i], g_syn[i], method=args.metric))
            # L1 norm of each layer gradient
            real_norm = torch.norm(g_real[i].reshape(g_real[i].shape[0], -1), p =1)
            syn_norm = torch.norm(g_syn[i].reshape(g_syn[i].shape[0], -1), p = 1)

            layer_grad_real.append(real_norm.detach().cpu())
            layer_grad_syn.append(syn_norm.detach().cpu())



    return loss, (layer_grad_real, layer_grad_syn)


def pretrain_sample(args, model, verbose=False):
    """Load pretrained networks
    """
    folder_base = f'./pretrained/{args.datatag}/{args.modeltag}_cut_pre'
    folder_list = glob.glob(f'{folder_base}*')
    tag = np.random.randint(len(folder_list))
    folder = folder_list[tag]

    epoch = args.pt_from
    if args.pt_num > 1:
        epoch = np.random.randint(args.pt_from, args.pt_from + args.pt_num)
    ckpt = f'checkpoint{epoch}.pth.tar'

    file_dir = os.path.join(folder, ckpt)
    load_ckpt(model, file_dir, verbose=verbose)

# def apply_pruning(args, model):
#     if args.pruning_type == "fc":
#         parameters_to_prune = [
#         (model.fc, 'weight')
#         ]
#     elif args.pruning_type == "global":
#         parameters_to_prune = [
#             (model.layer0.conv1, 'weight'),
#             (model.layer1[0].conv1, 'weight'),
#             (model.layer1[0].conv2, 'weight'),
#             (model.layer2[0].conv1, 'weight'),
#             (model.layer2[0].conv2, 'weight'),
#             (model.layer2[0].downsample[0], 'weight'),
#             (model.layer3[0].conv1, 'weight'),
#             (model.layer3[0].conv2, 'weight'),
#             (model.layer3[0].downsample[0], 'weight'),
#             (model.layer4[0].conv1, 'weight'),
#             (model.layer4[0].conv2, 'weight'),
#             (model.layer4[0].downsample[0], 'weight'),
#             (model.fc, 'weight'),
#         ]

#     for module, name in parameters_to_prune:
#         if name == 'weight':
#             prune.random_unstructured(module, name='weight', amount=args.pruning_ratio)
#         else:
#             prune.random_unstructured(module, name='bias', amount=args.pruning_ratio)

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

def condense(args, logger, device='cuda'):
    """Optimize condensed data
    """
    # Define subclass list and reverse mapping
    cls_from = args.phase * args.nclass_sub
    cls_to = (args.phase + 1) * args.nclass_sub
    subclass_list = np.arange(cls_from, cls_to)
    real_to_idx = {}
    for i, c in enumerate(subclass_list):
        real_to_idx[c] = i

    # Define real dataset and loader
    trainset, val_loader = load_resized_data(args, subclass_list)
    ## This load target subclass data on GPU memory
    ## One can modify ClassPartMemDataLoader to dynamically read subclass data without pre-loading
    ## Note, the entire class data is random sampled via the dataloader iterator (used for training networks)
    loader_real = ClassPartMemDataLoader(subclass_list,
                                         real_to_idx,
                                         trainset,
                                         batch_size=args.batch_real,
                                         num_workers=args.workers,
                                         shuffle=True,
                                         pin_memory=True,
                                         drop_last=True)

    nclass = trainset.nclass
    nch, hs, ws = trainset[0][0].shape

    # Define syn dataset
    synset = Synthesizer(args, args.nclass_sub, subclass_list, nch, hs, ws)
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
    #     synset.test(args, val_loader, logger, bench=False)

    # Data distillation
    optim_img = torch.optim.SGD(synset.parameters(), lr=args.lr_img, momentum=args.mom_img)
    ts = utils.TimeStamp(args.time)
    n_iter = args.niter * 100 // args.inner_loop
    it_log = n_iter // 50
    it_test = [100, 200, 400, 1000, 1500, n_iter]
    # it_test = [1, 2, 5, 10, 20, 50, 100, 200, 250, n_iter]
    logger(f"\nevaluation for {it_test} iteration")
    logger(f"\nStart condensing with {args.match} matching for {n_iter} iteration")

    args.fix_iter = max(1, args.fix_iter)

    
    model_pool = []
    # seeds = []

    # ## get all possible seeds _300_{seeds}_30_40
    # folder_name = os.path.join(args.pretrained_dir, f"*_60_70")
    # folders = glob.glob(folder_name)
    # for folder in folders:
    #     seeds.append(folder.split("/")[-1].split("_")[2])
    # random.shuffle(seeds)
    # logger(f"Seeds: {seeds}")

    # ## generate model pool: according to args.pool_number and args.sample_accrange,
    # ## random pick specific number of model from folder pretrained/{args.datatag}/{args.modeltag}_cut/_300_{random_seed}_{args.sample_accrange[0]}_{args.sample_accrange[1]}/
    # if args.pretrained_dir and args.pool_number:
    #     # pool_folder = os.path.join(args.pretrained_dir, f"model_pool_{args.pool_number}/{args.sample_accrange[0]}_{args.sample_accrange[1]}")
    #     pool_folder = ""
        
    #     if pool_folder != "" and os.path.isdir(pool_folder):
    #         filenames = glob.glob(os.path.join(pool_folder, "*.pth.tar"))
            
    #         for filename in filenames:
    #             model_pool.append(filename)
    #     else:
    #         if args.sample_accrange[1] - args.sample_accrange[0] == 0:
    #             for seed in seeds:
    #                 folder_name = os.path.join(args.pretrained_dir, f"*_{seed}_initial")
    #                 folder = glob.glob(folder_name)[0]
    #                 logger(f"folder : {folder}")
    #                 model_pool.append(random.sample(glob.glob(os.path.join(folder, "*.pth.tar")), 1)[0])
    #         elif len(args.sample_accrange) == 2 and args.distributed_num == None:
    #             print(args.pretrained_dir)
    #             number_of_range = (args.sample_accrange[1] - args.sample_accrange[0])//10
            
    #             ## use number_of_range(3) to evenly distribute pool_number(10), e.g [3,4,3]
    #             num_of_each_range = [args.pool_number//number_of_range for _ in range(number_of_range)] 
    #             for i in range(args.pool_number%number_of_range):
    #                 num_of_each_range[i] += 1
    #             random.shuffle(num_of_each_range)
    #             logger(f"num_of_each_range: {num_of_each_range}")
                
    #             dis_seeds = []
    #             for num in num_of_each_range:
    #                 dis_seeds.append(seeds[:num])
    #                 seeds = seeds[num:]

    #             for ith, seed_in_folders in enumerate(dis_seeds):
    #                 for seed in seed_in_folders:
    #                     folder_name = os.path.join(args.pretrained_dir, f"*_{seed}_{args.sample_accrange[0]+ith*10}_{args.sample_accrange[0]+ith*10+10}")

    #                     ## get folders by folder_name and sample args.pool_number of folders randomly
    #                     folder = glob.glob(folder_name)[0]
    #                     logger(f"folder : {folder}")
    #                     model_pool.append(random.sample(glob.glob(os.path.join(folder, "*.pth.tar")), 1)[0])
    #             # os.mkdir(pool_folder)
    #             # for file_ in model_pool:
    #             #     shutil.copy(file_, pool_folder)
                    
    #         elif args.distributed_num:
    #             number_of_range = len(args.sample_accrange)//2
    #             num_of_each_range = args.distributed_num
    #             dis_seeds = []
    #             for num in num_of_each_range:
    #                 dis_seeds.append(seeds[:num])
    #                 seeds = seeds[num:]

    #             for ith, seed_in_folders in enumerate(dis_seeds):
    #                 for seed in seed_in_folders:
    #                     folder_name = os.path.join(args.pretrained_dir, f"*_{seed}_{args.sample_accrange[0 + ith*2]}_{args.sample_accrange[1+ ith*2]}")

    #                     ## get folders by folder_name and sample args.pool_number of folders randomly
    #                     folder = glob.glob(folder_name)[0]
    #                     logger(f"folder : {folder}")
    #                     model_pool.append(random.sample(glob.glob(os.path.join(folder, "*.pth.tar")), 1)[0])

    #         else:
    #             number_of_range = len(args.sample_accrange)//2

    #             num_of_each_range = [args.pool_number//number_of_range for _ in range(number_of_range)] 
    #             for i in range(args.pool_number%number_of_range):
    #                 num_of_each_range[i] += 1
    #             random.shuffle(num_of_each_range)
    #             logger(f"num_of_each_range: {num_of_each_range}")
                
    #             dis_seeds = []
    #             for num in num_of_each_range:
    #                 dis_seeds.append(seeds[:num])
    #                 seeds = seeds[num:]

    #             for ith, seed_in_folders in enumerate(dis_seeds):
    #                 for seed in seed_in_folders:
    #                     folder_name = os.path.join(args.pretrained_dir, f"*_{seed}_{args.sample_accrange[0 + ith*2]}_{args.sample_accrange[1+ ith*2]}")

    #                     ## get folders by folder_name and sample args.pool_number of folders randomly
    #                     folder = glob.glob(folder_name)[0]
    #                     logger(f"folder : {folder}")
    #                     model_pool.append(random.sample(glob.glob(os.path.join(folder, "*.pth.tar")), 1)[0])
    #             os.mkdir(pool_folder)
    #             for file_ in model_pool:
    #                 shutil.copy(file_, pool_folder)
    # path_model_pool = os.path.join(args.pretrained_dir, f"{args.pool_number}")
    # path_model_pool = os.path.join(path_model_pool, f"{args.sample_accrange[0]}_{args.sample_accrange[1]}")
    # model_pool = glob.glob(os.path.join(path_model_pool, "*.pth.tar"))
    

    logger(f"model pool: {model_pool}")


    for it in range(n_iter):
        wandb.log({"Progress": it, "ot_epoch":it})
        if it % args.fix_iter == 0:
            model = define_model(args, nclass).to(device)
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
            elif args.pt_from >= 0:
                pretrain_sample(args, model)
            if args.early > 0:
                for _ in range(args.early):
                    train_epoch(args,
                                loader_real,
                                model,
                                criterion,
                                optim_net,
                                aug=aug_rand,
                                mixup=args.mixup_net)

        # ============== model augmentation =================
        if args.apply_pruning:
            apply_pruning(args, model)
        elif args.apply_dropout:
            model = add_dropout(model).to(device)
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
        loss_total = 0
        synset.data.data = torch.clamp(synset.data.data, min=0., max=1.)

        
        for ot in tqdm(range(args.inner_loop)):
            # # ======== save class gradient of each layer =============
            # gradient_layers_real = {}
            # gradient_layers_syn = {}

            # id2name = {}
            # id_ = 0
            # for name, layer in model.named_modules():
            #     if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
            #         gradient_layers_real[name] = []
            #         gradient_layers_syn[name] = []
            #         id2name[id_] = name
            #         id_ += 1
            
            # # ======== save class gradient of each layer =============

            ts.set()
            # Update synset
            inner_loss = 0
            for c in subclass_list:
                img, lab = loader_real.class_sample(c)
                img_syn, lab_syn = synset.sample(real_to_idx[c], max_size=args.batch_syn_max)
                ts.stamp("data")

                n = img.shape[0]
                img_aug = aug(torch.cat([img, img_syn]))
                ts.stamp("aug")

                loss, grad = matchloss(args, img_aug[:n], img_aug[n:], lab, lab_syn, model)

                inner_loss += loss.item()
                loss_total += loss.item()
                ts.stamp("loss")

                optim_img.zero_grad()
                loss.backward()
                optim_img.step()
                ts.stamp("backward")

            #     # ======== save gradient ===============
            #     for id_, grad_real in enumerate(grad[0]):
            #         gradient_layers_real[id2name[id_]].append(grad_real)
            #     for id_, grad_syn in enumerate(grad[1]):
            #         gradient_layers_syn[id2name[id_]].append(grad_syn)
            #     # ======== save gradient ===============

            # # log gradient of each layer as histogram to wandb
            # for name, grad_real in gradient_layers_real.items():
            #     wandb.log({f"gradient/real/{name}": wandb.Histogram(grad_real),  "inner_epoch": ot+it*args.inner_loop}) 
            #     if name == 'fc':
            #         print(grad_real)           
            # for name, grad_syn in gradient_layers_syn.items():
            #     wandb.log({f"gradient/syn/{name}": wandb.Histogram(grad_syn),  "inner_epoch": ot+it*args.inner_loop})

            # Net update
            if args.n_data > 0:
                loader_net = loader_real
                for _ in range(args.net_epoch):
                    top1, top5, losses = train_epoch(args,
                                loader_net,
                                model,
                                criterion,
                                optim_net,
                                n_data=args.n_data,
                                aug=aug_rand,
                                mixup=args.mixup_net)
                    
                    wandb.log({"InnerLoop/loss":inner_loss,
                               "Expert/inner/top1":top1,
                               "Expert/inner/top5":top5,
                               "Expert/inner/loss":losses,
                               "inner_epoch": ot+it*args.inner_loop
                            })
            else:
                wandb.log({
                    "innerLoop/loss":inner_loss,
                })
            ts.stamp("net update")

            if (ot + 1) % 10 == 0:
                ts.flush()

        # # Logging
        # wandb.log({"Expert/top1":top1, 
        #             "Expert/top5":top5, 
        #             "Expert/loss":losses, "ot_epoch":it})
        
        wandb.log({"MatchLoss":loss_total/nclass/args.inner_loop, "ot_epoch":it})

        
        if it % it_log == 0:
            logger(
                f"{utils.get_time()} (Iter {it:3d}) loss: {loss_total/args.nclass_sub/args.inner_loop:.1f}"
            )
        if (it + 1) in it_test:
            save_img(os.path.join(args.save_dir, f'img{it+1}.png'),
                     synset.data,
                     unnormalize=False,
                     dataname=args.dataset)
            torch.save(
                [synset.data.detach().cpu(), synset.targets.cpu()],
                os.path.join(args.save_dir, f'data{it+1}.pt'))
            print("img and data saved!")

            if not args.test:
                continue
                # test_result = synset.test(args, val_loader, logger)
                # if len(test_result) >1:
                #     wandb.log({"Test/ConvNet":test_result[0], 
                #             "Test/ResNet":test_result[1], "ot_epoch":it})
                # else:
                #     wandb.log({"Test/ConvNet":test_result[0], "ot_epoch":it})


if __name__ == '__main__':
    import shutil
    from misc.utils import Logger
    from argument import args
    import torch.backends.cudnn as cudnn
    import json
    import wandb

    cudnn.benchmark = True

    assert args.ipc > 0
    if args.nclass_sub < 0:
        raise AssertionError("Set number of subclasses! (args.nclass_sub)")
    if args.phase < 0:
        raise AssertionError("Set phase number! (args.phase)")

    if args.sample_accrange:
        if len(args.sample_accrange) ==2 :
            wandb.init(sync_tensorboard=False,
                project = "DatasetDistillation",
                job_type = "CleanRepo",
                config = args,
                name = f'IDC_dataset:{args.dataset}{args.nclass}_phase_{args.phase}_ipc_ipc:{args.ipc}_model:{args.modeltag}_match:{args.match}_strategy:{args.strategy}_softlabel:{args.smoothing}_temperature:{args.temperature}_early:{args.early}_ptrange:({args.pt_from} - {args.pt_from+args.pt_num})_sampleRange:({args.sample_accrange[0]} - {args.sample_accrange[1]})_poolnumber:{args.pool_number}_pretrained:{args.pretrained_dir.split("/")[-1]}_dropout:{args.apply_dropout}_rate:{args.dropout_rate}_pruning:{args.apply_pruning}({args.pruning_ratio})_pruning_type:{args.pruning_type}_tuneEpoch:{args.num_tune_subnetwork}_Masking:{args.apply_masking}_masking_type:{args.masking_type}')
        elif len(args.sample_accrange) ==4:
            if args.distributed_num:
                wandb.init(sync_tensorboard=False,
                project = "DatasetDistillation",
                job_type = "CleanRepo",
                config = args,
                name = f'IDC_dataset:{args.dataset}{args.nclass}_phase_{args.phase}_ipc_ipc:{args.ipc}_model:{args.modeltag}_match:{args.match}_strategy:{args.strategy}_softlabel:{args.smoothing}_temperature:{args.temperature}_early:{args.early}_ptrange:({args.pt_from} - {args.pt_from+args.pt_num})_sampleRange:({args.sample_accrange[0]} - {args.sample_accrange[1]} - {args.sample_accrange[2]} - {args.sample_accrange[3]})_poolnumber:{args.pool_number}_dis_num:{args.distributed_num}_dropout:{args.apply_dropout}_rate:{args.dropout_rate}_pruning:{args.apply_pruning}({args.pruning_ratio})_pruning_type:{args.pruning_type}_Masking:{args.apply_masking}_masking_type:{args.masking_type}')
            else:
                wandb.init(sync_tensorboard=False,
                project = "DatasetDistillation",
                job_type = "CleanRepo",
                config = args,
                name = f'IDC_dataset:{args.dataset}{args.nclass}_phase_{args.phase}_ipc_ipc:{args.ipc}_model:{args.modeltag}_match:{args.match}_strategy:{args.strategy}_softlabel:{args.smoothing}_temperature:{args.temperature}_early:{args.early}_ptrange:({args.pt_from} - {args.pt_from+args.pt_num})_sampleRange:({args.sample_accrange[0]} - {args.sample_accrange[1]} - {args.sample_accrange[2]} - {args.sample_accrange[3]})_poolnumber:{args.pool_number}_dropout:{args.apply_dropout}_rate:{args.dropout_rate}_pruning:{args.apply_pruning}({args.pruning_ratio})_pruning_type:{args.pruning_type}_Masking:{args.apply_masking}_masking_type:{args.masking_type}')
            


    else:
         wandb.init(sync_tensorboard=False,
                project = "DatasetDistillation",
                job_type = "CleanRepo",
                config = args,
                name = f'IDC_dataset:{args.dataset}{args.nclass}_phase_{args.phase}_ipc_ipc:{args.ipc}_model:{args.modeltag}_match:{args.match}_strategy:{args.strategy}_softlabel:{args.smoothing}_temperature:{args.temperature}_early:{args.early}_ptrange:({args.pt_from} - {args.pt_from+args.pt_num})_sampleRange:()_poolnumber:{args.pool_number}_dropout:{args.apply_dropout}_rate:{args.dropout_rate}_pruning:{args.apply_pruning}({args.pruning_ratio})_pruning_type:{args.pruning_type}_Masking:{args.apply_masking}_masking_type:{args.masking_type}')


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