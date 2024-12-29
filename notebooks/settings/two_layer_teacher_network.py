"""
This module implements methods to sample data from 2-layer teacher network on PyTorch.
The input layer of the teacher network is the identity, therefore the number of units equals 
the input dimension. The output layer weights are drawn from a Uniform({-1, 1}) distribution.
"""

import numpy, torch

def sample_output_weights(input_dimension):
    return numpy.random.choice([-1, 1], input_dimension)

def get_dataloader(input_dimension, sample_size, batch_size, output_weights=None, *args, **kwargs):
    true_output_weights = sample_output_weights(input_dimension) if output_weights is None else output_weights

    X = numpy.random.normal(size=(sample_size, input_dimension))
    y = numpy.matmul(X * (X > 0), true_output_weights)
    
    with torch.no_grad():
        tensor_X = torch.Tensor(X)
        tensor_y = torch.Tensor(y).unsqueeze(1)

    dataset = torch.utils.data.TensorDataset(tensor_X, tensor_y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    
    return (dataloader, true_output_weights) if output_weights is None else dataloader