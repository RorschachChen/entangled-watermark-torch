from .mnist import build as build_mnist
from .cifar import build as build_cifar
from .speechcmd import build as build_speechcmd


def build_dataset(image_set, args):
    if args.dataset == 'mnist' or args.dataset == 'fashion':
        return build_mnist(image_set, args)
    if 'cifar' in args.dataset:
        return build_cifar(image_set, args)
    if args.dataset == 'speechcmd':
        return build_speechcmd(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
