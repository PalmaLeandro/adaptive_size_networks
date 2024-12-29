"""
This module implements a method to sample data from the noisy XOR setting studied in https://arxiv.org/pdf/2202.07626.pdf.
The method implements an optional rotation to study axis aligned biases.
"""

import math, numpy, torch, scipy

CLASSES = 2

def get_dataloader(within_cluster_variance:float, input_dimension:int, sample_size:int, 
                   label_flipping:float=0., batch_size:int=1, rotation_matrix=None, clusters_per_class=2, 
                   normalize_inputs=False, scale_inputs=None, shift_inputs=None, generator=None, 
                   *args, **kwargs):
    samples_cluster = numpy.random.choice(list(range(CLASSES * clusters_per_class)), size=sample_size)
    inputs = labels = None
    for cluster in range(CLASSES * clusters_per_class):
        cluster_num_samples = len(numpy.where(samples_cluster == cluster)[0])
        cluster_mean = [0.] * input_dimension
        cluster_mean[0] = numpy.cos(2 * math.pi * cluster / (CLASSES * clusters_per_class))
        cluster_mean[1] = numpy.sin(2 * math.pi * cluster / (CLASSES * clusters_per_class))
        cluster_inputs = numpy.random.normal(scale=within_cluster_variance ** 0.5, size=cluster_num_samples * input_dimension)
        cluster_inputs = cluster_inputs.reshape(cluster_num_samples, input_dimension) + cluster_mean
        cluster_labels = numpy.repeat((cluster % CLASSES - 0.5) * 2., cluster_num_samples)
        inputs = cluster_inputs if inputs is None else numpy.concatenate([inputs, cluster_inputs])
        labels = cluster_labels if labels is None else numpy.concatenate([labels, cluster_labels])

    rotation_matrix_ = scipy.stats.special_ortho_group.rvs(input_dimension) if rotation_matrix is None else rotation_matrix
    inputs = numpy.matmul(inputs, rotation_matrix_)

    if label_flipping > 0:
        label_flipping_mask = numpy.random.choice([1., -1.], size=sample_size, p=[1 - label_flipping, label_flipping])
        labels *= label_flipping_mask

    # Transform -1, +1 labels to 0, 1 for cross entropy loss
    labels += 1.
    labels *= 0.5
    
    with torch.no_grad():
        tensor_X = torch.Tensor(inputs)
        tensor_y = torch.Tensor(labels)
    
    if normalize_inputs: tensor_X /= tensor_X.norm(dim=1).max()
    if scale_inputs: tensor_X *= scale_inputs
    if shift_inputs: tensor_X += torch.Tensor(shift_inputs)
    dataset = torch.utils.data.TensorDataset(tensor_X, tensor_y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True if batch_size < sample_size else False, 
                                             generator=generator)
    
    return (dataloader, rotation_matrix_) if rotation_matrix is None else dataloader