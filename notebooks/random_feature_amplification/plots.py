"""Implements plots for the random feature amplification experiment."""

import os, sys, numpy, torch, matplotlib.pyplot

sys.path.append(os.path.abspath(os.path.join('..')))  # Allow repository modules to be imported

from utils.plotting import draw_figure_into_canvas

fig, ((positive_neurons_norms_ax, positive_neurons_positive_cluster1_alignment_ax, positive_neurons_positive_cluster2_alignment_ax),
      (negative_neurons_norms_ax, negative_neurons_positive_cluster1_alignment_ax, negative_neurons_positive_cluster2_alignment_ax)) = matplotlib.pyplot.subplots(2, 3, figsize=(18, 12))

positive_neurons_norms_ax.set_title('Positive Neurons L2 norm', fontname='Times New Roman')
positive_neurons_positive_cluster1_alignment_ax.set_title('Positive Neurons µ+1 alignment', fontname='Times New Roman')
positive_neurons_positive_cluster2_alignment_ax.set_title('Positive Neurons µ+2 alignment', fontname='Times New Roman')
negative_neurons_norms_ax.set_title('Negative Neurons L2 norm', fontname='Times New Roman')
negative_neurons_positive_cluster1_alignment_ax.set_title('Negative Neurons µ+1 alignment', fontname='Times New Roman')
negative_neurons_positive_cluster2_alignment_ax.set_title('Negative Neurons µ+2 alignment', fontname='Times New Roman')

axis = (positive_neurons_norms_ax, positive_neurons_positive_cluster1_alignment_ax, positive_neurons_positive_cluster2_alignment_ax,
        negative_neurons_norms_ax, negative_neurons_positive_cluster1_alignment_ax, negative_neurons_positive_cluster2_alignment_ax)

for ax in axis: 
  ax.set_ylabel('% Neurons', fontname='Times New Roman'); ax.set_ylim(0, 0.2)
  for tick in ax.get_xticklabels(): tick.set_fontname('Times New Roman')
  for tick in ax.get_yticklabels(): tick.set_fontname('Times New Roman')

def histogram_bars(histogram_frequencies, histogram_bins):
    histogram_bins = histogram_bins.detach().cpu().numpy()[:-1]
    histogram_bins_pace = min(histogram_bins[1] - histogram_bins[0], 0.01)
    histogram_bins += histogram_bins_pace / 2.
    histogram_frequencies = histogram_frequencies.detach().cpu().numpy()
    return histogram_bins, histogram_frequencies / histogram_frequencies.sum(), histogram_bins_pace
 
def plot_norm_and_alignments_histograms(model=None, input_dimension=None, hidden_units=None, epoch=None, canvas=None, histogram_resolution=20, rotation_matrix=None, **kwargs):
    positive_cluster1_mean = numpy.array([1. if dimension == 1 else 0. for dimension in range(1, input_dimension + 1)])
    positive_cluster2_mean = numpy.array([1. if dimension == 2 else 0. for dimension in range(1, input_dimension + 1)])

    if rotation_matrix is not None:
      positive_cluster1_mean = numpy.matmul(rotation_matrix, positive_cluster1_mean)
      positive_cluster2_mean = numpy.matmul(rotation_matrix, positive_cluster2_mean)

    model_device = next(model.parameters()).device
    positive_cluster1_mean = torch.Tensor(positive_cluster1_mean).to(model_device)
    positive_cluster2_mean = torch.Tensor(positive_cluster2_mean).to(model_device)
    
    positive_neurons_weights = model.input_layer_weights.t()[:int(hidden_units / 2)]
    positive_neurons_norms_ax.bar(*histogram_bars(*torch.histogram(torch.norm(positive_neurons_weights, dim=1), histogram_resolution)), label=str(epoch), alpha=0.3)
    positive_neurons_positive_cluster1_alignment_histogram = histogram_bars(*torch.histogram(torch.matmul(positive_neurons_weights, positive_cluster1_mean), histogram_resolution))
    positive_neurons_positive_cluster1_alignment_ax.bar(*positive_neurons_positive_cluster1_alignment_histogram, zorder=2, alpha=0.3, label=str(epoch))
    positive_neurons_positive_cluster2_alignment_histogram = histogram_bars(*torch.histogram(torch.matmul(positive_neurons_weights, positive_cluster2_mean), histogram_resolution))
    positive_neurons_positive_cluster2_alignment_ax.bar(*positive_neurons_positive_cluster2_alignment_histogram, zorder=2, alpha=0.3, label=str(epoch))
    
    negative_neurons_weights = model.input_layer_weights.t()[int(hidden_units / 2):]
    negative_neurons_norms_ax.bar(*histogram_bars(*torch.histogram(torch.norm(negative_neurons_weights, dim=1), histogram_resolution)), label=str(epoch), alpha=0.3)
    negative_neurons_positive_cluster1_alignment_histogram = histogram_bars(*torch.histogram(torch.matmul(negative_neurons_weights, positive_cluster1_mean), histogram_resolution))
    negative_neurons_positive_cluster1_alignment_ax.bar(*negative_neurons_positive_cluster1_alignment_histogram, zorder=2, alpha=0.3, label=str(epoch))
    negative_neurons_positive_cluster2_alignment_histogram = histogram_bars(*torch.histogram(torch.matmul(negative_neurons_weights, positive_cluster2_mean), histogram_resolution))
    negative_neurons_positive_cluster2_alignment_ax.bar(*negative_neurons_positive_cluster2_alignment_histogram, zorder=2, alpha=0.3, label=str(epoch))
    
    for ax in axis: ax.legend(title='Epoch', prop=dict(family='Times New Roman'), title_fontproperties=dict(family='Times New Roman'))
    if canvas is not None: draw_figure_into_canvas(fig, canvas)

def plot_samples_and_activation(inputs, labels, model, rotation_matrix=None, input_dimension=None, input_domain_scale=2., activation_resolution=1000, **kwargs):
      _, ax = matplotlib.pyplot.subplots(figsize=(10, 10))

      domain_mesh = numpy.array([
            numpy.concatenate(dimension_repetition)
            for dimension_repetition
            in numpy.meshgrid(*([numpy.linspace(-input_domain_scale, input_domain_scale, activation_resolution)] * 2))
      ]).transpose()
      
      if input_dimension > 2:
        domain_mesh = numpy.concatenate([domain_mesh, numpy.repeat(numpy.repeat(0., input_dimension - 2)[numpy.newaxis, :], len(domain_mesh), axis=0)], axis=1)
      if rotation_matrix is not None: domain_mesh = numpy.matmul(domain_mesh, rotation_matrix)

      activations = ((torch.sign(model(torch.Tensor(domain_mesh))) + 1.) * 0.5).cpu().detach().numpy()
      ax.imshow(numpy.reshape(activations, (activation_resolution, activation_resolution)),
                origin='lower', cmap='gray', vmin=0, vmax=1, zorder=0,
                extent=[-input_domain_scale, input_domain_scale, -input_domain_scale, input_domain_scale])

      inputs_ = inputs if rotation_matrix is None else numpy.matmul(inputs, rotation_matrix.transpose())
      ax.scatter(inputs_[:, 0], inputs_[:, 1], c=labels)