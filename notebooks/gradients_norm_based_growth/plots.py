"""Implements plots for the effective_rank-based growth experiment."""

import os, sys, functools, math, numpy, torch, matplotlib.pyplot, matplotlib.cm

sys.path.append(os.path.abspath(os.path.join('..')))  # Allow repository modules to be imported

from utils.plotting import draw_figure_into_canvas, plot_experiment

FONT = 'Times New Roman'
HISTOGRAM_BINS = 50

def _plot_eigenspectrum_and_effective_rank(eigenspectrum, effective_ranks, iteration, area_color, eigenspectrum_ax, effective_rank_ax,
                                           eigenspectrum_distribution_log_scale=True):
    eigenspectrum = numpy.array(eigenspectrum)
    if eigenspectrum_distribution_log_scale:
        eigenspectrum = numpy.log(eigenspectrum)
        eigenspectrum -= min(eigenspectrum[eigenspectrum != - numpy.inf])
        eigenspectrum /= max(eigenspectrum)
        eigenspectrum[eigenspectrum == - numpy.inf] = 0.

    eigenspectrum_ax.fill_between(list(range(len(eigenspectrum))), [- iteration / 2,] * len(eigenspectrum), eigenspectrum - iteration / 2, 
                                  facecolor=area_color, edgecolor='w', alpha=0.5)
    k_star = effective_ranks.index(min(effective_ranks))
    effective_ranks = numpy.array(effective_ranks) / max(effective_ranks)
    effective_rank_ax.plot(list(range(len(effective_ranks))), effective_ranks - iteration / 2, c=area_color)
    effective_rank_ax.vlines(k_star, - iteration / 2, max(eigenspectrum) - iteration / 2, colors='grey', alpha=0.5)
    effective_rank_ax.text(k_star, - iteration / 2, f'{k_star:.0f}', fontdict={'family': FONT})

def histogram_bars(histogram_frequencies, histogram_bins):
    histogram_bins_pace = histogram_bins[1] - histogram_bins[0]
    histogram_bins += histogram_bins_pace / 2.
    return histogram_bins, histogram_frequencies / histogram_frequencies.sum(), histogram_bins_pace

def _plot_eigenspectrum_distribution(eigenspectrum, iteration, area_color, eigenspectrum_distribution_ax, 
                                     histogram_bins=HISTOGRAM_BINS, eigenspectrum_distribution_log_scale=True):
    eigenspectrum = numpy.array(eigenspectrum)
    if eigenspectrum_distribution_log_scale:
        zero_eigenvalues = eigenspectrum[eigenspectrum==0]
        nonzero_eigenvalues = eigenspectrum[eigenspectrum!=0]
        bins, frequency, _ = histogram_bars(*numpy.histogram(numpy.log(nonzero_eigenvalues), bins=histogram_bins))
        bins = numpy.concatenate([numpy.array([0]), numpy.exp(bins)])
        frequency = numpy.concatenate([numpy.array([len(zero_eigenvalues) / float(len(eigenspectrum))]), frequency])
        width = numpy.diff(bins)
        width[0] = min(width)
        eigenspectrum_distribution_ax.set_xscale('log')
    else:
        bins, frequency, width = histogram_bars(*numpy.histogram(eigenspectrum, bins=histogram_bins))
        width = numpy.diff(bins)

    frequency /= max(frequency)
    eigenspectrum_distribution_ax.fill_between(bins[:-1], [- iteration / 2,] * len(frequency), frequency - iteration / 2, width, 
                                               color=area_color, alpha=0.3)

def plot_eigenspectrum(run=None, epoch=0, epoch_frequency=100, eigenspectrum_distribution_log_scale=True, canvas=None, **kwargs):     
    if epoch % epoch_frequency != 0: return
    fig, ((rs_ax, rs_ax2), (eigenspectrum_distribution_ax, eigenspectrum_distribution_ax2)) = matplotlib.pyplot.subplots(2, 2, figsize=(16, 16))
    rs_ax.set_title('Eigenspectrum ∂L/∂activations', fontname=FONT)
    rs_ax.set_xlabel('eigenvalue', fontname=FONT)
    effective_rank_ax = rs_ax.twinx()
    rs_ax2.set_title('Eigenspectrum activations', fontname=FONT)
    rs_ax2.set_xlabel('eigenvalue', fontname=FONT)
    effective_rank_ax2 = rs_ax2.twinx()

    area_colors = matplotlib.cm.OrRd_r(numpy.linspace(.2, .6, len(run['eigenspectrum_d_L_d_sigma'])))
    for iteration, (eigenspectrum_d_L_d_sigma, r_k_d_L_d_sigma, eigenspectrum_sigma, r_k_sigma, area_color) in enumerate(
            zip(run['eigenspectrum_d_L_d_sigma'], run['r_k_d_L_d_sigma'], run['eigenspectrum_sigma'], run['r_k_sigma'], area_colors
        )):
       _plot_eigenspectrum_and_effective_rank(eigenspectrum_d_L_d_sigma, r_k_d_L_d_sigma, iteration, area_color, rs_ax, effective_rank_ax,
                                              eigenspectrum_distribution_log_scale=eigenspectrum_distribution_log_scale)
       _plot_eigenspectrum_and_effective_rank(eigenspectrum_sigma, r_k_sigma, iteration, area_color, rs_ax2, effective_rank_ax2,
                                              eigenspectrum_distribution_log_scale=eigenspectrum_distribution_log_scale)
       
       _plot_eigenspectrum_distribution(eigenspectrum_d_L_d_sigma, iteration, area_color, eigenspectrum_distribution_ax,
                                              eigenspectrum_distribution_log_scale=eigenspectrum_distribution_log_scale)
       _plot_eigenspectrum_distribution(eigenspectrum_sigma, iteration, area_color, eigenspectrum_distribution_ax2,
                                              eigenspectrum_distribution_log_scale=eigenspectrum_distribution_log_scale)

    for ax in (rs_ax, rs_ax2, effective_rank_ax, effective_rank_ax2, eigenspectrum_distribution_ax, eigenspectrum_distribution_ax2): 
        for tick in ax.get_xticklabels() + ax.get_yticklabels(): tick.set_fontname(FONT)
        ax.yaxis.set_major_locator(matplotlib.pyplot.NullLocator())

    if canvas is not None: draw_figure_into_canvas(fig, canvas)

def plot_eigenspectrum_on_canvas(canvas, **kwargs):
   return functools.partial(plot_eigenspectrum, canvas=canvas)

def plot_samples_and_model_activation(dataloader, model, rotation_matrix, activation_resolution=1000, epoch=1, epoch_frequency=1, 
                                      fig=None, ax=None, canvas=None, mean=None, scale=None, **kwargs):
    if epoch % epoch_frequency != 0: return
    _, (inputs, labels) = next(enumerate(dataloader))
    inputs_ = inputs if rotation_matrix is None else numpy.matmul(inputs, rotation_matrix.transpose())
    input_dimension = inputs.shape[-1]
    if scale is None:
        scale = max(numpy.abs(inputs_[:, 0]).max(), numpy.abs(inputs_[:, 1]).max())

    domain_mesh = numpy.array([
        numpy.concatenate(dimension_repetition)
        for dimension_repetition
        in numpy.meshgrid(*([numpy.linspace(-scale, scale, activation_resolution)] * 2))
    ]).transpose()

    if mean is not None:
        domain_mesh += numpy.array(mean)

    if input_dimension > 2:
        domain_mesh = numpy.concatenate(
            [domain_mesh, numpy.repeat(numpy.repeat(0., input_dimension - 2)[numpy.newaxis, :], len(domain_mesh), axis=0)], axis=1
        )
    
    domain_mesh = numpy.matmul(domain_mesh, rotation_matrix)
    activations = ((torch.sign(model(torch.Tensor(domain_mesh).to(model.device))) + 1.) * 0.5).cpu().detach().numpy()

    if fig is None: fig, ax = matplotlib.pyplot.subplots(figsize=(10, 10))
    else: ax = ax if ax is not None else fig.axes[0]
    
    ax.imshow(numpy.reshape(activations, (activation_resolution, activation_resolution)),
            origin='lower', cmap='gray', vmin=0, vmax=1, zorder=0,
            extent=[-scale, scale, -scale, scale])
    ax.scatter(inputs_[:, 0], inputs_[:, 1], c=labels)
    ax.tick_params(which='both', bottom=False, top=False, labelbottom=False)
    if canvas: draw_figure_into_canvas(fig, canvas)
    return fig
      
def plot_samples_and_model_activation_on_canvas(canvas):
    return functools.partial(plot_samples_and_model_activation, canvas=canvas)

def plot_samples_and_model_activation_and_neurons(dataloader, model, rotation_matrix, input_dimension=None, activation_resolution=1000, epoch=1,
                                      epoch_frequency=1, canvas=None, mean=[0., 0.], scale=None, **kwargs):
    if epoch % epoch_frequency != 0: return
    _, (inputs, labels) = next(enumerate(dataloader))
    inputs_ = inputs if rotation_matrix is None else numpy.matmul(inputs, rotation_matrix.transpose())

    fig, ax = matplotlib.pyplot.subplots(figsize=(10, 10))
    ax.scatter(inputs_[:, 0], inputs_[:, 1], c=labels)

    neurons_weights = model.input_layer.weight.detach().cpu().numpy()
    positive_neurons_weights = numpy.matmul(neurons_weights[:math.ceil(model.hidden_units / 2.)], rotation_matrix.transpose())
    negative_neurons_weights = numpy.matmul(neurons_weights[math.ceil(model.hidden_units / 2.):], rotation_matrix.transpose())
    ax.scatter(positive_neurons_weights[:, 0], positive_neurons_weights[:, 1], c='b', marker='+', s=500)
    ax.scatter(negative_neurons_weights[:, 0], negative_neurons_weights[:, 1], c='r', marker='_', s=500)
    ax.tick_params(which='both', bottom=False, top=False, labelbottom=False)

    if scale is not None:
        ax.set_xlim(mean[0] - scale, mean[0] + scale); ax.set_ylim(mean[1] - scale, mean[1] + scale)
    else:
        scale = max(inputs_[:, 0].max(), inputs_[:, 1].max(), 
                  positive_neurons_weights[:, 0].max(), positive_neurons_weights[:, 1].max(), 
                  negative_neurons_weights[:, 0].max(), negative_neurons_weights[:, 1].max())

    domain_mesh = numpy.array([
          numpy.concatenate(dimension_repetition)
          for dimension_repetition
          in numpy.meshgrid(*([numpy.linspace(-scale, scale, activation_resolution)] * 2))
    ]).transpose()

    mean = numpy.array(mean)
    domain_mesh += numpy.array(mean)
    
    if input_dimension > 2:
        domain_mesh = numpy.concatenate([domain_mesh, numpy.repeat(numpy.repeat(0., input_dimension - 2)[numpy.newaxis, :], len(domain_mesh), axis=0)], axis=1)
    
    domain_mesh = numpy.matmul(domain_mesh, rotation_matrix)
    activations = ((torch.sign(model(torch.Tensor(domain_mesh).to(model.device))) + 1.) * 0.5).cpu().detach().numpy()

    ax.imshow(numpy.reshape(activations, (activation_resolution, activation_resolution)),
              origin='lower', cmap='gray', vmin=0, vmax=1, zorder=0,
              extent=[mean[0] - scale, mean[0] + scale, mean[1] - scale, mean[1] + scale])
    if canvas: draw_figure_into_canvas(fig, canvas)
    return fig

def plot_samples_and_model_activation_and_neurons_hyperplanes(dataloader, model, rotation_matrix, 
                                                              activation_resolution=1000, epoch=1, epoch_frequency=1, 
                                                              canvas=None, mean=[0., 0.], scale=None, **kwargs):
    if epoch % epoch_frequency != 0: return
    mean = numpy.array(mean)
    inputs = []; labels = []
    for batch_inputs, batch_labels in dataloader: inputs.append(batch_inputs); labels.append(batch_labels)
    inputs, labels = torch.concatenate(inputs), torch.concatenate(labels)

    input_dimension = inputs.shape[-1]
    inputs_ = inputs if rotation_matrix is None else numpy.matmul(inputs, rotation_matrix.transpose())

    fig, ax = matplotlib.pyplot.subplots(figsize=(10, 10))
    ax.scatter(inputs_[:, 0], inputs_[:, 1], c=labels, label=labels)

    input_layer = model.layers[0]
    neurons_weights = numpy.matmul(input_layer.weight.detach().cpu().numpy(), rotation_matrix.transpose())
    number_of_neurons = len(neurons_weights)
    neurons_bias = input_layer.bias.detach().cpu().numpy() if input_layer.bias is not None else numpy.repeat(0., number_of_neurons)

    if scale is not None:
        ax.set_xlim(mean[0] - scale, mean[0] + scale); ax.set_ylim(mean[1] - scale, mean[1] + scale)
    else:
        scale = max(inputs_[:, 0].max(), inputs_[:, 1].max(), 
                    neurons_weights[:, 0].max() if len(neurons_weights) else 0, 
                    neurons_weights[:, 1].max() if len(neurons_weights) else 0)
    
    xs = numpy.linspace(mean[0] - scale, mean[0] + scale)
    if len(neurons_weights):
        neurons_ys = (neurons_weights[:, 0][:, None] * xs + neurons_bias[:, None]) / (- neurons_weights[:, 1][:, None])
        for hyperplane_ys, neuron_weights in zip(neurons_ys, neurons_weights):
            ax.plot(xs, hyperplane_ys, c='b')
            ax.arrow(xs[len(xs) // 2], hyperplane_ys[len(hyperplane_ys) // 2], neuron_weights[0], neuron_weights[1], 
                    width=0.01, color='b')

    domain_mesh = numpy.array([
          numpy.concatenate(dimension_repetition)
          for dimension_repetition
          in numpy.meshgrid(*([numpy.linspace(-scale, scale, activation_resolution)] * 2))
    ]).transpose()

    domain_mesh += numpy.array(mean)
    
    if input_dimension > 2:
        domain_mesh = numpy.concatenate([domain_mesh, numpy.repeat(numpy.repeat(0., input_dimension - 2)[numpy.newaxis, :], len(domain_mesh), axis=0)], axis=1)
    
    domain_mesh = numpy.matmul(domain_mesh, rotation_matrix)
    activations = ((torch.sign(model(torch.Tensor(domain_mesh).to(model.device))) + 1.) * 0.5).cpu().detach().numpy()

    ax.imshow(numpy.reshape(activations, (activation_resolution, activation_resolution)),
              origin='lower', cmap='gray', vmin=0, vmax=1, zorder=0,
              extent=[mean[0] - scale, mean[0] + scale, mean[1] - scale, mean[1] + scale])
    if canvas: draw_figure_into_canvas(fig, canvas)
    return fig
      
def plot_samples_and_model_activation_on_canvas(canvas):
    return functools.partial(plot_samples_and_model_activation, canvas=canvas)

def make_list_homogeneous_shape_vector(inhomogeneous_list, complete_with=numpy.nan, complete_to=None):
    max_length = complete_to or max(len(element) for element in inhomogeneous_list)
    return numpy.array([element + [complete_with] * (max_length - len(element)) for element in inhomogeneous_list])

def plot_gradients_norms(run, summary_frequency=1, canvas=None):
    fig, (gradients_norm_ax, residual_norm_ax, residual_std_ax) = matplotlib.pyplot.subplots(1, 3, figsize=(15, 5))

    gradients_norm_ax.set_title('SNR ( || E[ ∂L/∂u ] || / E[|| ∂L/∂u ||])')
    residual_norm_ax.set_title('Projected residual norm')
    residual_std_ax.set_title('Projected residual std')

    iterations = [iteration * summary_frequency for iteration in range(len(run['gradients_average_norm']))]
    max_depth = len(run['gradients_average_norm'][-1])
    max_width = len(run['gradients_average_norm'][-1][-1])
    gradients_average_norm = make_list_homogeneous_shape_vector(run['gradients_average_norm'], complete_to=max_depth)
    gradients_average_norm = numpy.where(gradients_average_norm==0, numpy.nan, gradients_average_norm)
    average_gradient_norm = make_list_homogeneous_shape_vector(run['average_gradient_norm'], complete_to=max_depth)
    average_gradient_norm = numpy.where(average_gradient_norm==0, numpy.nan, average_gradient_norm)
    #residual_projection_norm = make_list_homogeneous_shape_vector(run['residual_projection_norm'], complete_to=max_depth)
    #residual_projection_norm = numpy.where(residual_projection_norm==0, numpy.nan, residual_projection_norm)
    #residual_projection_upper_bounds = make_list_homogeneous_shape_vector(run['residual_projection_upper_bounds'], complete_to=max_depth)
    #residual_projection_upper_bounds = numpy.where(residual_projection_upper_bounds==0, numpy.nan, residual_projection_upper_bounds)
    #residual_projection_std = make_list_homogeneous_shape_vector(run['residual_projection_std'], complete_to=max_depth)
    #residual_projection_std = numpy.where(residual_projection_std==0, numpy.nan, residual_projection_std)

    colors = matplotlib.pyplot.cm.get_cmap('tab10', max_depth * max_width)
    for layer in range(max_depth):
        units_at_layer = len(gradients_average_norm[:, layer][-1])
        residual_norm_ax.plot(iterations, run['residual_norm'], label=f"R", linestyle='--')
        residual_norm_ax.hlines(run['residual_norm'][-1], 0, iterations[-1], linestyles=':', alpha=0.3)
        residual_norm_ax.text(0, run['residual_norm'][-1], f"{run['residual_norm'][-1]:.6f}", ha='left')

        for unit in range(units_at_layer):
            color = colors(unit + layer * units_at_layer)
            gradients_norm_ax.plot(iterations, average_gradient_norm[:, layer, unit], label=f'|| E[∂L/∂u] || l{layer}u{unit}', linestyle='-', color=color)
            gradients_norm_ax.hlines(average_gradient_norm[:, layer, unit][-1], 0, iterations[-1], colors=color, linestyles=':', alpha=0.3)
            gradients_norm_ax.text(0, average_gradient_norm[:, layer, unit][-1], f"{average_gradient_norm[:, layer, unit][-1]:.6f}", ha='left')

            gradients_norm_ax.plot(iterations, gradients_average_norm[:, layer, unit], label=f'E[|| ∂L/∂u ||] l{layer}u{unit}', linestyle='--', color=color)
            gradients_norm_ax.hlines(gradients_average_norm[:, layer, unit][-1], 0, iterations[-1], colors=color, linestyles=':', alpha=0.3)
            gradients_norm_ax.text(0, gradients_average_norm[:, layer, unit][-1], f"{gradients_average_norm[:, layer, unit][-1]:.6f}", ha='left')

            #residual_norm_ax.plot(iterations, residual_projection_norm[:, layer, unit], label=f"r' l{layer}u{unit}", linestyle='-', color=color)
            #residual_norm_ax.hlines(residual_projection_norm[:, layer, unit][-1], 0, iterations[-1], colors=color, linestyles=':', alpha=0.3)
            #residual_norm_ax.text(0, residual_projection_norm[:, layer, unit][-1], f"{residual_projection_norm[:, layer, unit][-1]:.6f}", ha='left')
#
            #residual_std_ax.plot(iterations, residual_projection_upper_bounds[:, layer, unit], label=f"r' std UB l{layer}u{unit}", linestyle='--', color=color)
            #residual_std_ax.hlines(residual_projection_upper_bounds[:, layer, unit][-1], 0, iterations[-1], colors=color, linestyles=':', alpha=0.3)
            #residual_std_ax.text(0, residual_projection_upper_bounds[:, layer, unit][-1], f"{residual_projection_upper_bounds[:, layer, unit][-1]:.6f}", ha='left')
#
            #residual_std_ax.plot(iterations, residual_projection_std[:, layer, unit], label=f"r' std l{layer}u{unit}", linestyle='-', color=color)
            #residual_std_ax.hlines(residual_projection_std[:, layer, unit][-1], 0, iterations[-1], colors=color, linestyles=':', alpha=0.3)
            #residual_std_ax.text(0, residual_projection_std[:, layer, unit][-1], f"{residual_projection_std[:, layer, unit][-1]:.6f}", ha='left')

    for ax in (gradients_norm_ax, residual_norm_ax, residual_std_ax): ax.legend()
    if canvas: draw_figure_into_canvas(fig, canvas)
    return fig