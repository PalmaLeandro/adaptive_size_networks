import numpy
from .gaussian_clusters import get_dataloader as get_gaussian_clusters_dataloader

def sphere_2d_epsilon_net(input_dimension:int, epsilon:float, radious:float=1., *args, **kwargs):
    assert 0 < epsilon <= 2
    angle = 2 * numpy.arcsin(epsilon / (2 * radious)) # chord formula
    number_of_nodes = int(2 * numpy.pi / (2 * angle)) + 1
    nodes = []
    for node_index in range(number_of_nodes):
        node = radious * numpy.array([numpy.sin(2 * numpy.pi * (node_index / number_of_nodes)), 
                                      numpy.cos(2 * numpy.pi * (node_index / number_of_nodes))])
        if input_dimension > 2:
            node = numpy.concatenate([node, numpy.zeros(input_dimension - 2)])

        nodes.append(node)

    return numpy.array(nodes)

def get_clusters_means_and_labels(input_dimension:int, epsilon:float, radious:float=1., classes=2, 
                              random_clusters_label_assignment=False, *args, **kwargs):
    clusters_means = sphere_2d_epsilon_net(input_dimension, epsilon, radious)
    clusters_labels = numpy.array([cluster % float(classes) for cluster in range(len(clusters_means))])

    if classes > 2:
        one_hot_clusters_labels = numpy.zeros((len(clusters_labels), int(max(clusters_labels) + 1)))
        one_hot_clusters_labels[numpy.arange(len(clusters_labels)), clusters_labels.astype(int)] = 1
        clusters_labels = one_hot_clusters_labels

    if random_clusters_label_assignment:
        numpy.random.shuffle(clusters_labels)

    return clusters_means, clusters_labels

def get_dataloader(*args, **kwargs):
    clusters_means, clusters_labels = get_clusters_means_and_labels(*args, **kwargs)
    return get_gaussian_clusters_dataloader(clusters_means=clusters_means, clusters_labels=clusters_labels, 
                                            *args, **kwargs)
