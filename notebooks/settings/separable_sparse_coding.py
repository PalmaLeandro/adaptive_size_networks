import math, numpy, torch, scipy

def get_dataloader(input_dimension:int, dictionary_items:int, sample_size:int, 
                   label_flipping:float=0., batch_size:int=1, normalize_inputs=False, 
                   scale_inputs=None, shift_inputs=None, generator=None, rotation_matrix=None,
                   *args, **kwargs):
    n_positive_atoms = int(math.ceil(dictionary_items / 2.))
    n_negative_atoms = dictionary_items - n_positive_atoms
    positive_atoms = numpy.abs(numpy.random.normal(size=n_positive_atoms * input_dimension).reshape(n_positive_atoms, input_dimension))
    positive_atoms /= numpy.linalg.norm(positive_atoms, axis=1)[:, None]
    negative_atoms = -numpy.abs(numpy.random.normal(size=n_negative_atoms * input_dimension).reshape(n_negative_atoms, input_dimension))
    negative_atoms /= numpy.linalg.norm(negative_atoms, axis=1)[:, None]
    
    possible_codes = [format(positive_code, f'#0{n_positive_atoms + 2}b')[2:] for positive_code in list(range(1, 2 ** n_positive_atoms))]
    positive_codes = numpy.random.choice(possible_codes, size=(math.ceil(sample_size / 2)))
    positive_codes = [[int(i) for i in positive_code] for positive_code in positive_codes]
    positive_codes /= numpy.linalg.norm(positive_codes, ord=1, axis=1)[:, None]
    positive_samples = positive_codes @ positive_atoms
    
    possible_codes = [format(negative_code, f'#0{n_negative_atoms + 2}b')[2:] for negative_code in list(range(1, 2 ** n_negative_atoms))]
    negative_codes = numpy.random.choice(possible_codes, size=(math.ceil(sample_size / 2)))
    negative_codes = [[int(i) for i in negative_code] for negative_code in negative_codes]
    negative_codes /= numpy.linalg.norm(negative_codes, ord=1, axis=1)[:, None]
    negative_samples = negative_codes @ negative_atoms
    
    inputs = numpy.concatenate([positive_samples, negative_samples])
    labels = numpy.concatenate([numpy.repeat(1., len(positive_samples)), numpy.repeat(-1., len(negative_samples))])
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