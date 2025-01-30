import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split


def dataset_loader(dataset_name, batch_size=64, num_workers=24):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    if dataset_name.lower() == 'cifar100':
        transform_cifar100 = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        full_train_dataset = torchvision.datasets.CIFAR100(root=root_dir, train=True, download=False,
                                                           transform=transform_cifar100)
        full_valid_dataset = torchvision.datasets.CIFAR100(root=root_dir, train=False, download=False,
                                                           transform=transform_cifar100)

    elif dataset_name.lower() == 'cifar10':
        transform_cifar10 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        full_train_dataset = torchvision.datasets.CIFAR10(root=root_dir, train=True, download=False,
                                                          transform=transform_cifar10)
        full_valid_dataset = torchvision.datasets.CIFAR10(root=root_dir, train=False, download=False,
                                                          transform=transform_cifar10)

    elif dataset_name.lower() == 'mnist':
        transform_mnist = transforms.Compose([
            transforms.Resize(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        full_train_dataset = torchvision.datasets.MNIST(root=root_dir, train=True, download=False,
                                                        transform=transform_mnist)
        full_valid_dataset = torchvision.datasets.MNIST(root=root_dir, train=False, download=False,
                                                        transform=transform_mnist)

    elif dataset_name.lower() == 'fashionmnist':
        transform_fashionMnist = transforms.Compose([
            transforms.Resize(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

        full_train_dataset = torchvision.datasets.FashionMNIST(root=root_dir, train=True, download=False,
                                                               transform=transform_fashionMnist)
        full_valid_dataset = torchvision.datasets.FashionMNIST(root=root_dir, train=False, download=False,
                                                               transform=transform_fashionMnist)

    elif dataset_name.lower() == 'flower102':
        transform_flower = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        full_train_dataset = torchvision.datasets.Flowers102(root=root_dir, split='train', download=False,
                                                             transform=transform_flower)
        full_valid_dataset = torchvision.datasets.Flowers102(root=root_dir, split='val', download=False,
                                                             transform=transform_flower)
    elif dataset_name.lower() == 'caltech256':
        transform_caltech = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        full_train_dataset = torchvision.datasets.ImageFolder(root_dir + "/256_objectcategories/256_ObjectCategories/",
                                                              transform=transform_caltech)
        train_size = int(0.8 * len(full_train_dataset))
        test_size = len(full_train_dataset) - train_size
        full_train_dataset, full_valid_dataset = random_split(full_train_dataset, [train_size, test_size])
    else:
        raise ValueError("Unsupported dataset. Please check the dataset name and try again.")

    trainloader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validloader = DataLoader(full_valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return trainloader, validloader
