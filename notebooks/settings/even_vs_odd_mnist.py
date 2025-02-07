"""
This module implements methods to sample data from MNIST dataset with labels 
indicating if the digit is even (0) or odd (1).
"""

import torch, torchvision

WIDTH, HEIGHT, CHANNELS, CLASSES, TRAIN_SAMPLES, TEST_SAMPLES = 28, 28, 1, 2, 60_000, 10_000
MEAN, STANDARD_DEVIATION = (0.1307), (0.3081)

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(MEAN, STANDARD_DEVIATION),
    torchvision.transforms.Lambda(lambda x: torch.flatten(x))
])

def train_dataloader(batch_size=None, data_path='./data/'):
    dataset = torchvision.datasets.MNIST(data_path, train=True, download=True, transform=transform)
    dataset.targets = torch.where(dataset.targets % 2 == 0, 0, 1)
    return torch.utils.data.DataLoader(dataset, batch_size or len(dataset), shuffle=True)

def test_dataloader(batch_size=None, data_path='./data/'):
    dataset = torchvision.datasets.MNIST(data_path, train=False, download=True, transform=transform)
    dataset.targets = torch.where(dataset.targets % 2 == 0, 0, 1)
    return torch.utils.data.DataLoader(dataset, batch_size or len(dataset), shuffle=True)
