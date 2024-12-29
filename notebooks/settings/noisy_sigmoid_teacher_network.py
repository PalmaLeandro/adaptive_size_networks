"""
This module implements a method to sample data from the setting studied in https://arxiv.org/pdf/2202.07626.pdf.
"""

import numpy, torch

def get_dataloader(input_dimension, sample_size, batch_size, noise_scale=0., output_weights=None, *args, **kwargs):
    true_output_weights = numpy.random.normal(size=(input_dimension, input_dimension)) if output_weights is None else output_weights

    X = numpy.random.normal(size=(sample_size, input_dimension))
    y = 1. / (1. + numpy.exp(-numpy.matmul(X, true_output_weights)))
    if noise_scale > 0:
        y += noise_scale * numpy.random.normal(size=(sample_size, input_dimension))
    
    with torch.no_grad():
        tensor_X, tensor_y = torch.Tensor(X), torch.Tensor(y)

    dataset = torch.utils.data.TensorDataset(tensor_X, tensor_y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    
    return (dataloader, true_output_weights) if output_weights is None else dataloader