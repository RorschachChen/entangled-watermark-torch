from torchvision import datasets, transforms


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def build(image_set, args):
    assert args.dataset in ['cifar10', 'cifar100']
    is_training = image_set == 'train'
    if args.dataset == 'cifar10':
        dataset = datasets.CIFAR10('data/cifar10/', train=is_training, transform=transform, download=True)
    elif args.dataset == 'cifar100':
        dataset = datasets.CIFAR100('data/cifar100/', train=is_training, transform=transform, download=True)
    return dataset