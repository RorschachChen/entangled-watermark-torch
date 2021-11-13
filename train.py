import argparse
import json
import os
import pickle
import random
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR10, CIFAR100
from torchvision.datasets.mnist import MNIST, FashionMNIST
from torchvision.transforms import transforms

from core.base_trainer import BaseTrainer
from core.evaluator import Evaluator
from engine import train_extract_model
from models import build_model
from core.trainer import Trainer



def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--batch_size', help='batch size', type=int, default=64)
    parser.add_argument('-c', '--config', default='configs/mnist.json', type=str)
    parser.add_argument('--ratio',
                        help='ratio of amount of legitimate data to watermarked data',
                        type=float, default=1.)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--epochs', help='epochs for training without watermarking', type=int, default=10)
    parser.add_argument('--w_epochs', help='epochs for training with watermarking', type=int, default=10)
    parser.add_argument('--dataset', help='mnist, fashion, speechcmd, cifar10, or cifar100', type=str,
                        default="cifar10")
    parser.add_argument('--model_type', help='2_conv, lstm, or resnet', type=str, default="2_conv")
    parser.add_argument('--metric', help='distance metric used in snnl, euclidean or cosine', type=str,
                        default="cosine")
    parser.add_argument('--factors', help='weight factor for snnl', nargs='+', type=float, default=[32, 32, 32])
    parser.add_argument('--temperatures', help='temperature for snnl', nargs='+', type=float, default=[1, 1, 1])
    parser.add_argument('--threshold', help='threshold for estimated false watermark rate, should be <= 1/num_class',
                        type=float, default=0.1)
    parser.add_argument('--maxiter', help='iter of perturb watermarked data with respect to snnl', type=int, default=10)
    parser.add_argument('--w_lr', help='learning rate for perturbing watermarked data', type=float, default=0.01)
    parser.add_argument('--t_lr', help='learning rate for temperature', type=float, default=0.1)
    parser.add_argument('--source', help='source class of watermark', type=int, default=1)
    parser.add_argument('--target', help='target class of watermark', type=int, default=7)
    parser.add_argument('--shuffle', type=int, default=0)

    parser.add_argument('--seed', help='random seed', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--default', help='whether to use default hyperparameter, 0 or 1', type=int, default=1)
    parser.add_argument('--layers', help='number of layers, only useful if model is resnet', type=int, default=18)
    parser.add_argument('--distribution', help='use in or out of distribution watermark', type=str, default='out')

    # dataset parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--output_dir', default='logs',
                        help='path where to save, empty for no saving')
    parser.add_argument('--resume', default=False, help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)
    return parser


def prepare_dataset(args, dataset=None):
    ds = args.dataset if dataset is None else dataset

    ##### shortcut ######
    is_cifar = 'cifar' in ds
    root = f'data/{ds}'
    print('Loading dataset: ' + ds)

    DATASET = {
        'cifar10': CIFAR10,
        'cifar100': CIFAR100,
        'mnist': MNIST,
        'fashion': FashionMNIST
    }[ds]

    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

    ##### train transform #####
    if not is_cifar:
        transform_list = [transforms.ToTensor()]
    else:
        transform_list = [transforms.RandomCrop(32, padding=4),
                          transforms.RandomHorizontalFlip(),
                          transforms.ToTensor(),
                          transforms.Normalize(mean, std)]

    train_transforms = transforms.Compose(transform_list)

    ##### test transform #####
    if not is_cifar:
        transform_list = [transforms.ToTensor()]
    else:
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean, std)]

    test_transforms = transforms.Compose(transform_list)

    ##### dataset and loader #####
    train_dataset = DATASET(root,
                            train=True,
                            transform=train_transforms,
                            download=True)
    test_dataset = DATASET(root,
                           train=False,
                           transform=test_transforms)

    # elif dataset == 'speechcmd':
    #     x_train = np.swapaxes(np.load(os.path.join(r"data", "sd_GSCmdV2", 'x_train.npy')), 1, 2)
    #     y_train = np.load(os.path.join(r"data", "sd_GSCmdV2", 'y_train.npy'))
    #     x_test = np.swapaxes(np.load(os.path.join(r"data", "sd_GSCmdV2", 'x_test.npy')), 1, 2)
    #     y_test = np.load(os.path.join(r"data", "sd_GSCmdV2", 'y_test.npy'))
    # else:
    #     raise NotImplementedError('Dataset is not implemented.')
    return train_dataset, test_dataset


def get_normal_loader(train_dataset, test_dataset):
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4,
                              drop_last=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size * 2,
                             shuffle=False,
                             num_workers=4)
    return train_loader, test_loader


def get_source_sampler(args, train_dataset):
    """
    根据in或out准备source data
    :return:
    """
    distribution = args.distribution
    dataset = args.dataset
    watermark_source = args.source
    # define the dataset and class to sample watermarked data
    if distribution == "in":
        dataset = train_dataset
    elif distribution == "out":
        if dataset == "mnist":
            w_dataset = "fashion"
        elif dataset == "fashion":
            w_dataset = "mnist"
        # elif "cifar" in dataset:
        # elif dataset == "speechcmd":
        else:
            raise NotImplementedError("OOD dataset")
        dataset, _ = prepare_dataset(args, w_dataset)
    else:
        raise NotImplementedError("Distribution could only be either \'in\' or \'out\'.")
    source_sampler = get_sampler(dataset, watermark_source)
    return source_sampler


def resume(model, optimizer):
    """
    从checkpoint恢复
    :param model:
    :param optimizer:
    :return:
    """
    if args.resume.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.resume, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    if not args.eval and 'optimizer' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1
    return [model, optimizer]


def get_transform(x_train, x_test):
    transform_list = []
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    transform_list.extend([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_transforms = transforms.Compose(transform_list)
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    for i in range(len(x_train)):
        x_train[i] = train_transforms(np.float32(x_train[i]))
    for i in range(len(x_test)):
        x_test[i] = train_transforms(np.float32(x_test[i]))
    return x_train, x_test


class SourceSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, mask, data_source):
        self.mask = mask
        self.data_source = data_source

    def __iter__(self):
        return iter([i.item() for i in torch.nonzero(self.mask)])

    def __len__(self):
        return len(self.data_source)


def main(args):
    # loading default configs
    config = json.load(open(args.config))
    if args.default:
        for conf in config:
            setattr(args, conf, config[conf])
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ewe_model, extract_model, plain_model = build_model(args)
    ewe_model.to(device)
    extract_model.to(device)
    plain_model.to(device)

    # where optimizer linked with model parameters
    ewe_optimizer = torch.optim.Adam(ewe_model.parameters(), lr=args.lr, betas=(0.9, 0.99),
                                     weight_decay=args.weight_decay)
    extract_optimizer = torch.optim.Adam(extract_model.parameters(), lr=args.lr, betas=(0.9, 0.99),
                                         weight_decay=args.weight_decay)
    clean_optimizer = torch.optim.Adam(plain_model.parameters(), lr=args.lr, betas=(0.9, 0.99),
                                       weight_decay=args.weight_decay)

    if args.resume:
        ewe_model, ewe_optimizer = resume(ewe_model, ewe_optimizer)
        extract_model, extract_optimizer = resume(ewe_model, extract_optimizer)
        plain_model, clean_optimizer = resume(ewe_model, clean_optimizer)

    # 用Dataset加载的包含aug，用numpy加载的没有aug
    train_dataset, test_dataset = prepare_dataset(args)
    train_loader, test_loader = get_normal_loader(train_dataset, test_dataset)

    x_train, y_train, x_test, y_test = prepare_dataset_np(args)
    source_data = get_source_data(args.distribution, x_train, y_train, args.source, args.dataset)
    target_data = x_train[y_train == args.target]
    # make sure watermarked data is the same size as target data
    trigger = np.concatenate([source_data] * (target_data.shape[0] // source_data.shape[0] + 1), 0)[
              :target_data.shape[0]]
    w_num_batch = target_data.shape[0] // args.batch_size * 2
    ewe_trainer = Trainer(args, device, ewe_model, ewe_optimizer, train_loader)
    watermark_evaluator = Evaluator(args, test_loader, trigger, w_num_batch, args.output_channel, device)
    # watermark_evaluator.evaluate(ewe_model)
    watermark_evaluator.evaluate(extract_model, False)
    watermark_evaluator.evaluate(plain_model, False)
    start_time = time.time()
    ewe_trainer.normal_train()
    print(f"ewe_model normal train finished in {time.time() - start_time}s. ")
    start_time = time.time()
    ewe_trainer.fgsm_optimize_trigger(trigger, target_data, w_num_batch)
    print(f"optimize trigger finished in {time.time() - start_time}s. ")
    num_batch = x_train.shape[0] // args.batch_size
    start_time = time.time()
    ewe_trainer.watermark_model(num_batch, w_num_batch, trigger, target_data)
    print(f"watermark finished in {time.time() - start_time}s. ")
    victim_acc, victim_watermark_acc = watermark_evaluator.evaluate(ewe_model)
    print(f"Victim Model || validation accuracy: {victim_acc}, "
          f"watermark success: {victim_watermark_acc}")
    # 训练extract model
    start_time = time.time()
    train_extract_model(extract_model, ewe_model, train_loader, extract_optimizer, args.epochs + args.w_epochs, device)
    print(f"train extract model finished in {time.time() - start_time}s. ")
    extracted_acc, extracted_watermark_acc = watermark_evaluator.evaluate(extract_model, False)
    print(f"Extracted Model || validation accuracy: {extracted_acc}, "
          f"watermark success: {extracted_watermark_acc}")
    # 训练clean model
    start_time = time.time()
    clean_model_trainer = BaseTrainer(args.epochs + args.w_epochs, plain_model, clean_optimizer, train_loader, device)
    clean_model_trainer.normal_train(False)
    print(f"train clean model finished in {time.time() - start_time}")
    clean_acc, clean_watermark_acc = watermark_evaluator.evaluate(plain_model, False)
    print(f"Clean Model || validation accuracy: {clean_acc}, "
          f"watermark success: {clean_watermark_acc}")


def get_source_data(distribution, x_train, y_train, source, dataset):
    height = x_train[0].shape[0]
    width = x_train[0].shape[1]
    try:
        channels = x_train[0].shape[2]
    except:
        channels = 1
    # define the dataset and class to sample watermarked data
    if distribution == "in":
        source_data = x_train[y_train == source]
    elif distribution == "out":
        if dataset == "mnist":
            w_dataset = "fashion"
            with open(os.path.join("data", f"{w_dataset}.pkl"), 'rb') as f:
                w_data = pickle.load(f)
            x_w, y_w = w_data["training_images"], w_data["training_labels"]
        elif dataset == "fashion":
            w_dataset = "mnist"
            with open(os.path.join("data", f"{w_dataset}.pkl"), 'rb') as f:
                w_data = pickle.load(f)
            x_w, y_w = w_data["training_images"], w_data["training_labels"]
        elif "cifar" in dataset:
            import scipy.io as sio
            w_dataset = sio.loadmat(os.path.join("data", "train_32x32"))
            x_w, y_w = np.moveaxis(w_dataset['X'], -1, 0), np.squeeze(w_dataset['y'] - 1)
        elif dataset == "speechcmd":
            x_w = np.swapaxes(np.load(os.path.join(r"data", "sd_GSCmdV2", 'trigger.npy')), 1, 2)
            y_w = np.ones(x_w.shape[0]) * source
        else:
            raise NotImplementedError()
        x_w = np.reshape(x_w / 255, [-1, height, width, channels])
        source_data = x_w[y_w == source]
    else:
        raise NotImplementedError("Distribution could only be either \'in\' or \'out\'.")
    return source_data


def prepare_dataset_np(args):
    dataset = args.dataset
    if dataset == 'mnist' or dataset == 'fashion':
        with open(os.path.join("data", f"{dataset}.pkl"), 'rb') as f:
            mnist = pickle.load(f)
        x_train, y_train, x_test, y_test = mnist["training_images"], mnist["training_labels"], \
                                           mnist["test_images"], mnist["test_labels"]
        x_train = np.reshape(x_train / 255, [-1, 28, 28, 1])
        x_test = np.reshape(x_test / 255, [-1, 28, 28, 1])
    elif "cifar" in dataset:
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader
        transform_list = []
        transform_list.extend([
            transforms.ToTensor(),
        ])
        train_transforms = transforms.Compose(transform_list)
        test_transforms = transforms.Compose(transform_list)
        if dataset == 'cifar10':
            train_dataset = datasets.CIFAR10('./data/CIFAR10/', train=True, transform=train_transforms, download=True)
            test_dataset = datasets.CIFAR10('./data/CIFAR10/', train=False, transform=test_transforms, download=True)
        else:
            train_dataset = datasets.CIFAR100('./data/CIFAR100/', train=True, transform=train_transforms, download=True)
            test_dataset = datasets.CIFAR100('./data/CIFAR100/', train=False, transform=test_transforms, download=True)
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
        train_pair = next(iter(train_loader))
        train_dataset_array = train_pair[0].permute(0, 2, 3, 1).numpy()
        test_pair = next(iter(test_loader))
        test_dataset_array = test_pair[0].permute(0, 2, 3, 1).numpy()
        x_train = train_dataset_array
        y_train = train_pair[1].numpy()
        x_test = test_dataset_array
        y_test = test_pair[1].numpy()
    elif dataset == 'speechcmd':
        x_train = np.swapaxes(np.load(os.path.join(r"data", "sd_GSCmdV2", 'x_train.npy')), 1, 2)
        y_train = np.load(os.path.join(r"data", "sd_GSCmdV2", 'y_train.npy'))
        x_test = np.swapaxes(np.load(os.path.join(r"data", "sd_GSCmdV2", 'x_test.npy')), 1, 2)
        y_test = np.load(os.path.join(r"data", "sd_GSCmdV2", 'y_test.npy'))
    else:
        raise NotImplementedError('Dataset is not implemented.')
    return x_train, y_train, x_test, y_test


def get_sampler(train_dataset, watermark_source):
    mask = [1 if train_dataset[i][1] == watermark_source else 0 for i in range(len(train_dataset))]
    mask = torch.tensor(mask)
    sampler = SourceSampler(mask, train_dataset)
    return sampler


if __name__ == '__main__':
    parser = argparse.ArgumentParser('training process of entangle implemented in Pytorch', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
