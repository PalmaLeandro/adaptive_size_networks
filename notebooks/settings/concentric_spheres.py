"""
This module implements a method to sample data from a distribution of two concentric spheres.
"""

import numpy, scipy, torch

CLASSES = 2

def get_dataset(input_dimension:int, sample_size:int, spheres_dimension:int=None, spheres:int=CLASSES, 
                margin:float=0.1, mean=None, rotation_matrix=None, noise_std=1., label_flipping=0., *args, **kwargs):
    spheres_dimension = spheres_dimension if spheres_dimension is not None and spheres_dimension < input_dimension else input_dimension
    spheres_indices = list(range(spheres))
    samples_spheres_indices = torch.tensor(numpy.random.choice(spheres_indices, size=sample_size), dtype=torch.float32).unsqueeze(1)
    inputs = torch.normal(mean=0, std=1, size=(sample_size, spheres_dimension), dtype=torch.float32)
    inputs = torch.nn.functional.normalize(inputs)
    inputs -= inputs * samples_spheres_indices * margin
    inputs += torch.normal(mean=0, std=noise_std, size=(sample_size, spheres_dimension), dtype=torch.float32)
    labels = samples_spheres_indices % CLASSES

    if label_flipping > 0:
        # Transform 0, 1 labels to -1, 1 for label flipping
        labels *= 2.
        labels -= 1.
        label_flipping_mask = torch.tensor(
            numpy.random.choice([1., -1.], size=sample_size, p=[1 - label_flipping, label_flipping])
        ).unsqueeze(1)
        labels *= label_flipping_mask
        # Transform -1, +1 labels to 0, 1 for cross entropy loss
        labels += 1.
        labels *= 0.5

    if mean is not None: inputs += mean
    
    extra_dimensions = input_dimension - spheres_dimension
    if extra_dimensions:
        inputs = torch.cat([inputs, torch.normal(mean=0, std=noise_std, size=(sample_size, extra_dimensions), dtype=torch.float32)], dim=1)

    rotation_matrix_ = scipy.stats.special_ortho_group.rvs(input_dimension) if rotation_matrix is None else rotation_matrix
    inputs = inputs.mm(torch.tensor(rotation_matrix_, dtype=torch.float32))
    dataset = torch.utils.data.TensorDataset(inputs, labels.squeeze())
    return (dataset, rotation_matrix_) if rotation_matrix is None else dataset

def get_dataloader(sample_size:int, batch_size:int=None, rotation_matrix=None, generator=None, *args, **kwargs):
    if rotation_matrix is None:
        dataset, rotation_matrix_ = get_dataset(*args, sample_size=sample_size, **kwargs)    
    else:
        dataset = get_dataset(*args, sample_size=sample_size, rotation_matrix=rotation_matrix, **kwargs)    
        
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=sample_size if batch_size is None else batch_size, 
                                             shuffle=True if batch_size < sample_size else False, generator=generator)
    return (dataloader, rotation_matrix_) if rotation_matrix is None else dataloader
