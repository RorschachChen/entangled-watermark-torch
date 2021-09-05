from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


def build(image_set, args):
    assert args.dataset in ['fashion', 'mnist']
    is_training = image_set == 'train'
    if args.dataset == 'fashion':
        dataset = datasets.FashionMNIST('data/fashion/', train=is_training, transform=transform, download=True)
    elif args.dataset == 'mnist':
        dataset = datasets.MNIST('data/mnist/', train=is_training, transform=transform, download=True)
    return dataset
