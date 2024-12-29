import numpy, torch, scipy

def get_dataloader(input_dimension:int, sample_size:int, clusters_means, clusters_labels, classes, within_cluster_variance:float=0, 
                   label_flipping:float=0., batch_size:int=1, rotation_matrix=None, 
                   normalize_inputs=False, scale_inputs=None, shift_inputs=None, balance_classes=False, generator=None, 
                   *args, **kwargs):
    if sample_size == len(clusters_means) and within_cluster_variance == 0:
        samples_cluster = numpy.array(list(range(len(clusters_means))))
    else:
        samples_cluster = numpy.random.choice(list(range(len(clusters_means))), size=sample_size)

    inputs = labels = None
    classes_samples_count = [0, ] * classes
    unique_labels = numpy.unique(clusters_labels, axis=0 if classes > 2 else None).tolist()
    for cluster, (cluster_mean, cluster_label) in enumerate(zip(clusters_means, clusters_labels)):
        cluster_num_samples = len(numpy.where(samples_cluster == cluster)[0])
        cluster_inputs = numpy.random.normal(scale=within_cluster_variance ** 0.5, size=cluster_num_samples * input_dimension)
        cluster_inputs = cluster_inputs.reshape(cluster_num_samples, input_dimension) + numpy.array(cluster_mean)
        cluster_labels = numpy.array([cluster_label,] * cluster_num_samples)
        inputs = cluster_inputs if inputs is None else numpy.concatenate([inputs, cluster_inputs])
        labels = cluster_labels if labels is None else numpy.concatenate([labels, cluster_labels])
        cluster_label_index = numpy.argwhere((unique_labels == cluster_label).all(1)).flatten()[0] if classes > 2 else int(cluster_label)
        classes_samples_count[cluster_label_index] += cluster_num_samples

    rotation_matrix_ = scipy.stats.special_ortho_group.rvs(input_dimension) if rotation_matrix is None else rotation_matrix
    if normalize_inputs: inputs /= numpy.linalg.norm(inputs, axis=1)[:, numpy.newaxis]
    if scale_inputs: inputs *= scale_inputs
    if shift_inputs: inputs += numpy.array(shift_inputs)
    inputs = numpy.matmul(inputs, rotation_matrix_)
    if label_flipping > 0:
        label_flipping_mask = numpy.random.choice([0, 1], size=sample_size, p=[1 - label_flipping, label_flipping])
        random_labels = numpy.random.choice(unique_labels, size=sample_size, p=[(1. / len(unique_labels))])
        labels = numpy.where(label_flipping_mask, labels, random_labels)
    
    if balance_classes:
        most_frequent_class_frequency = max(classes_samples_count)
        for class_index, class_samples_count in zip(range(classes), classes_samples_count):
            if class_samples_count < most_frequent_class_frequency:
                less_frequent_class_samples_count = classes_samples_count[class_index]
                assert less_frequent_class_samples_count > 0, 'Less frequent class sample count needs to be > 0'

                missing_samples = (most_frequent_class_frequency - less_frequent_class_samples_count)
                less_frequent_class_sampels_indices = (labels == unique_labels[class_index]).nonzero()[0]
                samples_to_add_indices = numpy.random.choice(less_frequent_class_sampels_indices, size=missing_samples)
                samples_to_add = numpy.copy(inputs[samples_to_add_indices])
                inputs = numpy.concatenate([inputs, samples_to_add])
                labels = numpy.concatenate([labels, [unique_labels[class_index],] * missing_samples])
                
    with torch.no_grad():
        tensor_X = torch.Tensor(inputs)
        tensor_y = torch.Tensor(labels)

    dataset = torch.utils.data.TensorDataset(tensor_X, tensor_y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True if batch_size < sample_size else False, 
                                             generator=generator)
    
    return (dataloader, rotation_matrix_) if rotation_matrix is None else dataloader