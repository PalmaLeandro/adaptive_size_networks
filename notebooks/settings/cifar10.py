"""
This module implements methods to sample data from CIFAR10 dataset.
"""

import logging, torch, torchvision

WIDTH, HEIGHT, CHANNELS, CLASSES, SAMPLE_SIZE = 32, 32, 3, 10, 50000
MEANS, STANDARD_DEVIATIONS = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)

def train_dataloader(batch_size, data_path='./data/'):
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(MEANS, STANDARD_DEVIATIONS),
    ])

    dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=train_transforms)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

def test_dataloader(batch_size, data_path='./data/'):
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(MEANS, STANDARD_DEVIATIONS),
    ])

    dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=test_transforms)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)
