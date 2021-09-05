import argparse
import json
import os
import pickle
import random
import time
from pathlib import Path
import tensorflow as tf
import numpy as np
import torch
import util.misc as utils
from engine import train_model, train_ewe_model
from models import build_model
from torchvision.transforms import transforms

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--batch_size', help='batch size', type=int, default=512)
    parser.add_argument('-c', '--config', default='configs/mnist.json', type=str)
    parser.add_argument('--ratio',
                        help='ratio of amount of legitimate data to watermarked data',
                        type=float, default=1.)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
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
    parser.add_argument('--seed', help='random seed', type=int, default=0)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def augment_train(image):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_with_crop_or_pad(image, 40, 40)
    image = tf.image.random_crop(image, size=[image.shape[0], 32, 32, 3])
    image = tf.image.random_flip_left_right(image)
    random_angles = tf.random.uniform(shape=(tf.shape(image)[0],), minval=-np.pi / 8, maxval=np.pi / 8)
    image = tf.contrib.image.transform(image, tf.contrib.image.angles_to_projective_transforms(
            random_angles, 32, 32))
    image = tf.image.per_image_standardization(image)
    sess = tf.Session()
    with sess.as_default():
        ret = image.eval()
    sess.close()
    return ret


def augment_test(image):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    sess = tf.Session()
    with sess.as_default():
        ret = image.eval()
    sess.close()
    return ret


def prepare_dataset(args):
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
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        transform_list = []
        transform_list.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        train_transforms = transforms.Compose(transform_list)
        transform_list = []
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
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
        # BHWC
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

    # if "cifar" in dataset:
    #     x_train = augment_train(x_train)
    #     x_test = augment_test(x_test)
        # x_train, x_test = get_transform(x_train, x_test)
    return x_train, y_train, x_test, y_test


def prepare_source_data(args, x_train, y_train):
    """
    根据in或out准备source data
    :return:
    """
    distribution = args.distribution
    dataset = args.dataset
    watermark_source = args.source
    # define the dataset and class to sample watermarked data
    height = x_train[0].shape[0]
    width = x_train[0].shape[1]
    try:
        channels = x_train[0].shape[2]
    except:
        channels = 1
    if distribution == "in":
        source_data = x_train[y_train == watermark_source]
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
            y_w = np.ones(x_w.shape[0]) * watermark_source
        else:
            raise NotImplementedError()
        x_w = np.reshape(x_w / 255, [-1, height, width, channels])
        source_data = x_w[y_w == watermark_source]
    else:
        raise NotImplementedError("Distribution could only be either \'in\' or \'out\'.")
    return source_data


def resume(model, optimizer, lr_scheduler):
    """
    从checkpoint恢复
    :param model:
    :param optimizer:
    :param lr_scheduler:
    :return:
    """
    if args.resume.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.resume, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
    return [model, optimizer, lr_scheduler]


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


def main(args):
    # loading default configs
    config = json.load(open(args.config))
    if args.default:
        for conf in config:
            setattr(args, conf, config[conf])

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if 'cifar100' == args.dataset:
        args.input_channel = 3
        args.output_channel = 100
    elif 'cifar10' == args.dataset:
        args.input_channel = 3
        args.output_channel = 100
    elif 'fashion' == args.dataset or 'mnist' == args.dataset:
        args.input_channel = 1
        args.output_channel = 10
    else:
        raise NotImplementedError()

    ewe_model, extract_model, plain_model = build_model(args)
    ewe_model.to(device)
    extract_model.to(device)
    plain_model.to(device)

    # where optimizer linked with model parameters
    ewe_optimizer = torch.optim.AdamW(ewe_model.parameters(), lr=args.lr, betas=(0.9, 0.99),
                                      weight_decay=args.weight_decay)
    extract_optimizer = torch.optim.AdamW(extract_model.parameters(), lr=args.lr, betas=(0.9, 0.99),
                                          weight_decay=args.weight_decay)
    clean_optimizer = torch.optim.AdamW(plain_model.parameters(), lr=args.lr, betas=(0.9, 0.99),
                                        weight_decay=args.weight_decay)
    # Decays the learning rate of each parameter group by gamma every step_size epochs
    ewe_lr_scheduler = torch.optim.lr_scheduler.StepLR(ewe_optimizer, args.lr_drop)
    extract_lr_scheduler = torch.optim.lr_scheduler.StepLR(extract_optimizer, args.lr_drop)
    clean_lr_scheduler = torch.optim.lr_scheduler.StepLR(clean_optimizer, args.lr_drop)

    x_train, y_train, x_test, y_test = prepare_dataset(args)
    source_data = prepare_source_data(args, x_train, y_train)
    watermark_source = args.source
    target_data = x_train[y_train == watermark_source]
    # make sure watermarked data is the same size as target data
    trigger = np.concatenate([source_data] * (target_data.shape[0] // source_data.shape[0] + 1), 0)[
              :target_data.shape[0]]
    num_class = len(np.unique(y_train))
    batch_size = args.batch_size
    trigger_label = np.zeros([batch_size, num_class])
    trigger_label[:, watermark_source] = 1
    w_num_batch = target_data.shape[0] // batch_size * 2

    output_dir = Path(args.output_dir)
    if args.resume:
        ewe_model, ewe_optimizer, ewe_lr_scheduler = resume(ewe_model, ewe_optimizer, ewe_lr_scheduler)
        extract_model, extract_optimizer, extract_lr_scheduler = resume(ewe_model, extract_optimizer,
                                                                        extract_lr_scheduler)
        plain_model, clean_optimizer, clean_lr_scheduler = resume(ewe_model, clean_optimizer, clean_lr_scheduler)
    # 训练ewe_model
    # x_train, y_train, args, target_data, trigger, model, optimizer, device
    train_ewe_model(x_train, y_train, x_test, y_test, args, target_data, trigger, ewe_model, ewe_optimizer, device, "ewe")
    # ewe训练好的model的输出作为label
    extracted_data, extracted_label = prepare_target_for_extract(ewe_model, x_train, batch_size, device)
    # 训练extract model
    train_model(args, extracted_data, extracted_label, x_test, y_test, extract_model, extract_optimizer, extract_lr_scheduler,
                'extract', device, batch_size, output_dir, w_num_batch, trigger)
    # 重头训练clean model
    train_model(args, x_train, y_train, x_test, y_test, plain_model, clean_optimizer, clean_lr_scheduler, 'clean',
                device, batch_size, output_dir, w_num_batch, trigger)


def prepare_target_for_extract(model, x_train, batch_size, device):
    extracted_label = []
    num_batch = x_train.shape[0] // batch_size
    for batch in range(num_batch):
        input = torch.from_numpy(x_train[batch * batch_size: (batch + 1) * batch_size]).to(device, torch.float)
        output = model(input)
        extracted_label.append(output.argmax(dim=1).cpu().numpy())
    return x_train, np.concatenate(extracted_label, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('training process of entangle implemented in Pytorch', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
