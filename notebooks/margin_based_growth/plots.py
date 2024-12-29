import os, sys, numpy, pandas, torch, matplotlib.pyplot, matplotlib.cm

sys.path.append(os.path.abspath(os.path.join('..')))  # Allow repository modules to be imported

from utils.plots import draw_figure_into_canvas

HISTOGRAM_BINS = 50

def plot_series_and_reference_on_ax(ax, x, y, label, color=None, linestyle=None):
    ax.plot(x, y, c=color, label=label, linestyle=linestyle)
    ax.hlines(y[-1], 0, x[-1], color='gray', alpha=.3, linestyle='--')
    ax.text(0, y[-1], f'{y[-1]:.5f}', ha='left')


def plot_train_loss_and_accuracy(train_loss, test_loss, train_accuracy, test_accuracy, sample_size, model_metrics, 
                                 batch_size=None, canvas=None, *args, **kwargs):
    if canvas: matplotlib.pyplot.ioff()
    fig, (loss_ax, accuracy_ax) = matplotlib.pyplot.subplots(1, 2, figsize=(16, 8))
    loss_ax.set_title('Train Loss')
    loss_ax.set_xlabel('iterations')
    accuracy_ax.set_title('Accuracy')
    accuracy_ax.set_xlabel('iterations')
    epochs = len(train_loss)
    batch_size = sample_size if batch_size is None else batch_size
    iterations = [iteration * sample_size / batch_size for iteration in range(epochs)]
    plot_series_and_reference_on_ax(loss_ax, iterations, train_loss, 'train', 'b')
    plot_series_and_reference_on_ax(loss_ax, iterations, test_loss, 'test', 'r')
    plot_series_and_reference_on_ax(accuracy_ax, iterations, train_accuracy, 'train', 'b')
    plot_series_and_reference_on_ax(accuracy_ax, iterations, test_accuracy, 'test', 'r')
    #plot_architecture_changes_to_ax(loss_ax, sample_size, batch_size, model_metrics)
    #plot_architecture_changes_to_ax(accuracy_ax, sample_size, batch_size, model_metrics)
    for ax in (loss_ax, accuracy_ax): ax.legend()
    loss_ax.set_yscale('log')
    if canvas: draw_figure_into_canvas(fig, canvas)

def histogram_bars(histogram_frequencies, histogram_bins):
    histogram_bins = histogram_bins.detach().cpu().numpy()[:-1]
    histogram_bins_pace = histogram_bins[1] - histogram_bins[0]
    histogram_bins += histogram_bins_pace / 2.
    histogram_frequencies = histogram_frequencies.detach().cpu().numpy()
    return histogram_bins, histogram_frequencies / histogram_frequencies.sum(), histogram_bins_pace

def plot_model_metrics(model_metrics, batch_size, sample_size, target_accuracy, canvas=None, *args, **kwargs):
    _, (margins_histogram_ax, positive_margins_ax, gradient_norms_ax) = matplotlib.pyplot.subplots(1, 3, figsize=(18, 6))
    margins_histogram_ax.set_title('Margins')
    margins_histogram_ax.set_xlabel('margin (< ∂L/∂w, w > / || w || )')
    margins_histogram_ax.set_ylabel('% samples')
    positive_margins_ax.set_title('Growth metrics')
    positive_margins_ax.set_xlabel('iteration')
    positive_margins_ax.set_ylabel('samples')
    gradient_norms_ax.set_title('Gradient norms')
    gradient_norms_ax.set_xlabel('iteration')
    gradient_norms_ax.set_ylabel('L2 norm')

    model_metrics_df = pandas.DataFrame(model_metrics)
    model_metrics_df['iterations'] = model_metrics_df['epoch'] * sample_size / batch_size
    colors = matplotlib.pyplot.cm.get_cmap('tab10', max(model_metrics_df['layer'].max(), 10))
    model_metrics_df['most_unstable_neuron'] = model_metrics_df.apply(
        lambda row: (numpy.array(row['positive_margins_count']) / numpy.array(row['active_samples_count'])).argmin(axis=0),
        axis='columns'
    )
    for metric in ['active_samples_count', 'positive_margins_count', 'gradients_average_norm', 'average_gradient_norm', 'margins']:
        model_metrics_df[metric] = model_metrics_df.apply(lambda row: row[metric][row['most_unstable_neuron']], axis='columns')

    for layer, layer_df in model_metrics_df.groupby('layer'):
        margins_histogram_ax.bar(*histogram_bars(*torch.histogram(torch.tensor(layer_df['margins'].iloc[-1]), HISTOGRAM_BINS)), 
                                 color=colors(layer), label=layer, alpha=0.3)
        positive_margins_ax.plot(layer_df['iterations'].tolist(), (layer_df['positive_margins_count']).to_numpy(), 
                                 label=f'positive samples {layer}', marker='+', c=colors(layer))
        positive_margins_ax.plot(layer_df['iterations'].tolist(), (layer_df['active_samples_count']).to_numpy() * target_accuracy, 
                                 label=f'threshold {layer}', linestyle='--', c=colors(layer))
        positive_margins_ax.plot(layer_df['iterations'].tolist(), (layer_df['samples_not_captured_count']).to_numpy(), 
                                 label=f'samples not captured {layer}', marker='o', c=colors(layer))
        plot_series_and_reference_on_ax(gradient_norms_ax, layer_df['iterations'].tolist(), 
                                        layer_df['gradients_average_norm'].replace(0, numpy.nan).tolist(), f'E[|| ∂L/∂u ||] {layer}', 
                                        color=colors(layer), linestyle='-')
        plot_series_and_reference_on_ax(gradient_norms_ax, layer_df['iterations'].tolist(), 
                                        layer_df['average_gradient_norm'].replace(0, numpy.nan).tolist(), f'|| E[∂L/∂u] || {layer}', 
                                        color=colors(layer), linestyle='--')

def plot_weights_and_biases_gradient_norms(model_metrics, batch_size, sample_size, canvas=None, *args, **kwargs):
    fig, (weights_gradient_norms_ax, bias_gradient_norms_ax) = matplotlib.pyplot.subplots(1, 2, figsize=(16, 8))
    weights_gradient_norms_ax.set_title('Weights gradient norms')
    weights_gradient_norms_ax.set_xlabel('iteration')
    weights_gradient_norms_ax.set_ylabel('L2 norm')
    bias_gradient_norms_ax.set_title('Bias gradient norms')
    bias_gradient_norms_ax.set_xlabel('iteration')
    bias_gradient_norms_ax.set_ylabel('L2 norm')

    model_metrics_df = pandas.DataFrame(model_metrics)
    model_metrics_df['iterations'] = model_metrics_df['epoch'] * sample_size / batch_size
    colors = matplotlib.pyplot.cm.get_cmap('tab10', max(model_metrics_df['layer'].max(), 10))
    model_metrics_df['most_unstable_neuron'] = model_metrics_df.apply(
        lambda row: (numpy.array(row['positive_margins_count']) / numpy.array(row['active_samples_count'])).argmin(axis=0),
        axis='columns'
    )
    for metric in ['active_samples_count', 'positive_margins_count', 'gradients_average_norm', 'average_gradient_norm', 'margins']:
        model_metrics_df[metric] = model_metrics_df.apply(lambda row: row[metric][row['most_unstable_neuron']], axis='columns')

    for layer, layer_df in model_metrics_df.groupby('layer'):
        plot_series_and_reference_on_ax(weights_gradient_norms_ax, layer_df['iterations'].tolist(), 
                                        layer_df['gradients_average_norm'].replace(0, numpy.nan).tolist(), f'E[|| ∂L/∂w ||] {layer}', 
                                        color=colors(layer), linestyle='-')
        plot_series_and_reference_on_ax(weights_gradient_norms_ax, layer_df['iterations'].tolist(), 
                                        layer_df['average_gradient_norm'].replace(0, numpy.nan).tolist(), f'|| E[∂L/∂w] || {layer}', 
                                        color=colors(layer), linestyle='--')
        plot_series_and_reference_on_ax(bias_gradient_norms_ax, layer_df['iterations'].tolist(), 
                                        layer_df['pre_activation_gradient_average_norm'].replace(0, numpy.nan).tolist(), f'E[|| ∂L/∂u ||] {layer}', 
                                        color=colors(layer), linestyle='-')
        plot_series_and_reference_on_ax(bias_gradient_norms_ax, layer_df['iterations'].tolist(), 
                                        layer_df['pre_activation_average_gradient_norm'].replace(0, numpy.nan).tolist(), f'|| E[∂L/∂u] || {layer}', 
                                        color=colors(layer), linestyle='--')

    #plot_architecture_changes_to_ax(weights_gradient_norms_ax, sample_size, batch_size, model_metrics)
    #plot_architecture_changes_to_ax(weights_gradient_norms_ax, sample_size, batch_size, model_metrics)
    for ax in (weights_gradient_norms_ax, bias_gradient_norms_ax): ax.legend(title='Layer'); ax.set_yscale('log')
    if canvas: draw_figure_into_canvas(fig, canvas)

def plot_samples_activation_hyperplanes(dataloader, model, rotation_matrix, activation_resolution=1000, epoch=1, epoch_frequency=1, 
                                        canvas=None, mean=[0., 0.], scale=None, **kwargs):
    if epoch % epoch_frequency != 0: return
    mean = numpy.array(mean)
    inputs = []; labels = []
    for batch_inputs, batch_labels in dataloader: inputs.append(batch_inputs); labels.append(batch_labels)
    inputs, labels = torch.concatenate(inputs), torch.concatenate(labels)

    input_dimension = inputs.shape[-1]
    inputs_ = inputs if rotation_matrix is None else numpy.matmul(inputs, rotation_matrix.transpose())

    fig, ax = matplotlib.pyplot.subplots(figsize=(10, 10))
    ax.scatter(inputs_[:, 0], inputs_[:, 1], c=labels, label=labels, cmap=matplotlib.cm.get_cmap('RdYlBu'))

    input_layer = model.layers[0]
    neurons_weights = numpy.matmul(input_layer.weight.detach().cpu().numpy(), rotation_matrix.transpose())
    number_of_neurons = len(neurons_weights)
    neurons_bias = input_layer.bias.detach().cpu().numpy() if input_layer.bias is not None else numpy.repeat(0., number_of_neurons)

    if scale is not None:
        ax.set_xlim(mean[0] - scale, mean[0] + scale); ax.set_ylim(mean[1] - scale, mean[1] + scale)
    else:
        scale = max(inputs_[:, 0].max(), inputs_[:, 1].max())
    
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
        extra_dimensions = numpy.repeat(numpy.repeat(0., input_dimension - 2)[numpy.newaxis, :], len(domain_mesh), axis=0)
        domain_mesh = numpy.concatenate([domain_mesh, extra_dimensions], axis=1)
    
    domain_mesh = numpy.matmul(domain_mesh, rotation_matrix)
    activations = ((torch.sign(model(torch.Tensor(domain_mesh).to(model.device))) + 1.) * 0.5).cpu().detach().numpy()

    ax.imshow(numpy.reshape(activations, (activation_resolution, activation_resolution)),
              origin='lower', cmap='gray', vmin=0, vmax=1, zorder=0,
              extent=[mean[0] - scale, mean[0] + scale, mean[1] - scale, mean[1] + scale])
    if canvas: draw_figure_into_canvas(fig, canvas)
    return fig

def plot_weights_norm_and_biases(model_metrics, batch_size, sample_size, canvas=None, *args, **kwargs):
    fig, (weights_gradient_norms_ax, bias_gradient_norms_ax) = matplotlib.pyplot.subplots(1, 2, figsize=(16, 8))
    weights_gradient_norms_ax.set_title('Weights gradient norms')
    weights_gradient_norms_ax.set_xlabel('iteration')
    weights_gradient_norms_ax.set_ylabel('L2 norm')
    bias_gradient_norms_ax.set_title('Bias gradient norms')
    bias_gradient_norms_ax.set_xlabel('iteration')
    bias_gradient_norms_ax.set_ylabel('L2 norm')

    model_metrics_df = pandas.DataFrame(model_metrics)
    model_metrics_df['iterations'] = model_metrics_df['epoch'] * sample_size / batch_size
    colors = matplotlib.pyplot.cm.get_cmap('tab10', max(model_metrics_df['layer'].max(), 10))
    model_metrics_df['most_unstable_neuron'] = model_metrics_df.apply(
        lambda row: (numpy.array(row['positive_margins_count']) / numpy.array(row['active_samples_count'])).argmin(axis=0),
        axis='columns'
    )
    for metric in ['active_samples_count', 'positive_margins_count', 'gradients_average_norm', 'average_gradient_norm', 'margins']:
        model_metrics_df[metric] = model_metrics_df.apply(lambda row: row[metric][row['most_unstable_neuron']], axis='columns')

    for layer, layer_df in model_metrics_df.groupby('layer'):
        plot_series_and_reference_on_ax(weights_gradient_norms_ax, layer_df['iterations'].tolist(), 
                                        layer_df['gradients_average_norm'].replace(0, numpy.nan).tolist(), f'E[|| ∂L/∂w ||] {layer}', 
                                        color=colors(layer), linestyle='-')
        plot_series_and_reference_on_ax(weights_gradient_norms_ax, layer_df['iterations'].tolist(), 
                                        layer_df['average_gradient_norm'].replace(0, numpy.nan).tolist(), f'|| E[∂L/∂w] || {layer}', 
                                        color=colors(layer), linestyle='--')
        plot_series_and_reference_on_ax(bias_gradient_norms_ax, layer_df['iterations'].tolist(), 
                                        layer_df['pre_activation_gradient_average_norm'].replace(0, numpy.nan).tolist(), f'E[|| ∂L/∂u ||] {layer}', 
                                        color=colors(layer), linestyle='-')
        plot_series_and_reference_on_ax(bias_gradient_norms_ax, layer_df['iterations'].tolist(), 
                                        layer_df['pre_activation_average_gradient_norm'].replace(0, numpy.nan).tolist(), f'|| E[∂L/∂u] || {layer}', 
                                        color=colors(layer), linestyle='--')

    #plot_architecture_changes_to_ax(weights_gradient_norms_ax, sample_size, batch_size, model_metrics)
    #plot_architecture_changes_to_ax(weights_gradient_norms_ax, sample_size, batch_size, model_metrics)
    for ax in (weights_gradient_norms_ax, bias_gradient_norms_ax): ax.legend(title='Layer'); ax.set_yscale('log')
    if canvas: draw_figure_into_canvas(fig, canvas)



def plot_model_metrics(model_metrics, batch_size, sample_size, target_accuracy, canvas=None, save_figure_path=None, figure_name_parameters=None, *args, **kwargs):
    fig, (margins_histogram_ax, positive_margins_ax, gradient_norms_ax) = matplotlib.pyplot.subplots(1, 3, figsize=(18, 6))
    margins_histogram_ax.set_title('Margins')
    margins_histogram_ax.set_xlabel('margin (< ∂L/∂w, w > / || w || )')
    margins_histogram_ax.set_ylabel('% samples')
    positive_margins_ax.set_title('Growth metrics')
    positive_margins_ax.set_xlabel('iteration')
    positive_margins_ax.set_ylabel('samples')
    gradient_norms_ax.set_title('Gradient norms')
    gradient_norms_ax.set_xlabel('iteration')
    gradient_norms_ax.set_ylabel('L2 norm')

    model_metrics_df = pandas.DataFrame(model_metrics)
    model_metrics_df['iterations'] = model_metrics_df['epoch'] * sample_size / batch_size
    colors = matplotlib.pyplot.cm.get_cmap('tab10', max(model_metrics_df['layer'].max(), 10))
    model_metrics_df['most_unstable_neuron'] = model_metrics_df.apply(
        lambda row: (numpy.array(row['positive_margins_count']) / numpy.array(row['active_samples_count'])).argmin(axis=0),
        axis='columns'
    )
    for metric in ['active_samples_count', 'positive_margins_count', 'gradients_average_norm', 'average_gradient_norm', 'margins']:
        model_metrics_df[metric] = model_metrics_df.apply(lambda row: row[metric][row['most_unstable_neuron']], axis='columns')

    for layer, layer_df in model_metrics_df.groupby('layer'):
        margins_histogram_ax.bar(*histogram_bars(*torch.histogram(torch.tensor(layer_df['margins'].iloc[-1]), HISTOGRAM_BINS)), 
                                 color=colors(layer), label=layer, alpha=0.3)
        positive_margins_ax.plot(layer_df['iterations'].tolist(), (layer_df['positive_margins_count']).to_numpy(), 
                                 label=f'positive samples {layer}', marker='+', c=colors(layer))
        positive_margins_ax.plot(layer_df['iterations'].tolist(), (layer_df['active_samples_count']).to_numpy() * target_accuracy, 
                                 label=f'threshold {layer}', linestyle='--', c=colors(layer))
        positive_margins_ax.plot(layer_df['iterations'].tolist(), (layer_df['samples_not_captured_count']).to_numpy(), 
                                 label=f'samples not captured {layer}', marker='o', c=colors(layer))
        plot_series_and_reference_on_ax(gradient_norms_ax, layer_df['iterations'].tolist(), 
                                        layer_df['gradients_average_norm'].replace(0, numpy.nan).tolist(), f'E[|| ∂L/∂u ||] {layer}', 
                                        color=colors(layer), linestyle='-')
        plot_series_and_reference_on_ax(gradient_norms_ax, layer_df['iterations'].tolist(), 
                                        layer_df['average_gradient_norm'].replace(0, numpy.nan).tolist(), f'|| E[∂L/∂u] || {layer}', 
                                        color=colors(layer), linestyle='--')
    
    if canvas: draw_figure_into_canvas(fig, canvas)
    if save_figure_path: save_figure(fig, {'batch_size': batch_size, 'sample_size': sample_size, **kwargs}, save_figure_path, figure_name_parameters, 'model_metrics')

def plot_weights_and_biases_gradient_norms(model_metrics, batch_size, sample_size, canvas=None, 
                                           save_figure_path=None, figure_name_parameters=None, *args, **kwargs):
    fig, (weights_gradient_norms_ax, bias_gradient_norms_ax) = matplotlib.pyplot.subplots(1, 2, figsize=(16, 8))
    weights_gradient_norms_ax.set_title('Weights gradient norms')
    weights_gradient_norms_ax.set_xlabel('iteration')
    weights_gradient_norms_ax.set_ylabel('L2 norm')
    bias_gradient_norms_ax.set_title('Bias gradient norms')
    bias_gradient_norms_ax.set_xlabel('iteration')
    bias_gradient_norms_ax.set_ylabel('L2 norm')

    model_metrics_df = pandas.DataFrame(model_metrics)
    model_metrics_df['iterations'] = model_metrics_df['epoch'] * sample_size / batch_size
    colors = matplotlib.pyplot.cm.get_cmap('tab10', max(model_metrics_df['layer'].max(), 10))
    model_metrics_df['most_unstable_neuron'] = model_metrics_df.apply(
        lambda row: (numpy.array(row['positive_margins_count']) / numpy.array(row['active_samples_count'])).argmin(axis=0),
        axis='columns'
    )
    for metric in ['active_samples_count', 'positive_margins_count', 'gradients_average_norm', 'average_gradient_norm', 'margins']:
        model_metrics_df[metric] = model_metrics_df.apply(lambda row: row[metric][row['most_unstable_neuron']], axis='columns')

    for layer, layer_df in model_metrics_df.groupby('layer'):
        plot_series_and_reference_on_ax(weights_gradient_norms_ax, layer_df['iterations'].tolist(), 
                                        layer_df['gradients_average_norm'].replace(0, numpy.nan).tolist(), f'E[|| ∂L/∂w ||] {layer}', 
                                        color=colors(layer), linestyle='-')
        plot_series_and_reference_on_ax(weights_gradient_norms_ax, layer_df['iterations'].tolist(), 
                                        layer_df['average_gradient_norm'].replace(0, numpy.nan).tolist(), f'|| E[∂L/∂w] || {layer}', 
                                        color=colors(layer), linestyle='--')
        plot_series_and_reference_on_ax(bias_gradient_norms_ax, layer_df['iterations'].tolist(), 
                                        layer_df['pre_activation_gradient_average_norm'].replace(0, numpy.nan).tolist(), f'E[|| ∂L/∂u ||] {layer}', 
                                        color=colors(layer), linestyle='-')
        plot_series_and_reference_on_ax(bias_gradient_norms_ax, layer_df['iterations'].tolist(), 
                                        layer_df['pre_activation_average_gradient_norm'].replace(0, numpy.nan).tolist(), f'|| E[∂L/∂u] || {layer}', 
                                        color=colors(layer), linestyle='--')

    plot_architecture_changes_to_ax(weights_gradient_norms_ax, sample_size, batch_size, model_metrics)
    plot_architecture_changes_to_ax(weights_gradient_norms_ax, sample_size, batch_size, model_metrics)
    for ax in (weights_gradient_norms_ax, bias_gradient_norms_ax): ax.legend(title='Layer'); ax.set_yscale('log')
    if canvas: draw_figure_into_canvas(fig, canvas)
    if save_figure_path: save_figure(fig, {'batch_size': batch_size, 'sample_size': sample_size, **kwargs}, save_figure_path, figure_name_parameters, 'weights_and_biases_gradient_norms')



def plot_weights_norm_and_biases(model_metrics, batch_size, sample_size, canvas=None, 
                                 save_figure_path=None, figure_name_parameters=None, *args, **kwargs):
    fig, (weights_gradient_norms_ax, bias_gradient_norms_ax) = matplotlib.pyplot.subplots(1, 2, figsize=(16, 8))
    weights_gradient_norms_ax.set_title('Weights gradient norms')
    weights_gradient_norms_ax.set_xlabel('iteration')
    weights_gradient_norms_ax.set_ylabel('L2 norm')
    bias_gradient_norms_ax.set_title('Bias gradient norms')
    bias_gradient_norms_ax.set_xlabel('iteration')
    bias_gradient_norms_ax.set_ylabel('L2 norm')

    model_metrics_df = pandas.DataFrame(model_metrics)
    model_metrics_df['iterations'] = model_metrics_df['epoch'] * sample_size / batch_size
    colors = matplotlib.pyplot.cm.get_cmap('tab10', max(model_metrics_df['layer'].max(), 10))
    model_metrics_df['most_unstable_neuron'] = model_metrics_df.apply(
        lambda row: (numpy.array(row['positive_margins_count']) / numpy.array(row['active_samples_count'])).argmin(axis=0),
        axis='columns'
    )
    for metric in ['active_samples_count', 'positive_margins_count', 'gradients_average_norm', 'average_gradient_norm', 'margins']:
        model_metrics_df[metric] = model_metrics_df.apply(lambda row: row[metric][row['most_unstable_neuron']], axis='columns')

    for layer, layer_df in model_metrics_df.groupby('layer'):
        plot_series_and_reference_on_ax(weights_gradient_norms_ax, layer_df['iterations'].tolist(), 
                                        layer_df['gradients_average_norm'].replace(0, numpy.nan).tolist(), f'E[|| ∂L/∂w ||] {layer}', 
                                        color=colors(layer), linestyle='-')
        plot_series_and_reference_on_ax(weights_gradient_norms_ax, layer_df['iterations'].tolist(), 
                                        layer_df['average_gradient_norm'].replace(0, numpy.nan).tolist(), f'|| E[∂L/∂w] || {layer}', 
                                        color=colors(layer), linestyle='--')
        plot_series_and_reference_on_ax(bias_gradient_norms_ax, layer_df['iterations'].tolist(), 
                                        layer_df['pre_activation_gradient_average_norm'].replace(0, numpy.nan).tolist(), f'E[|| ∂L/∂u ||] {layer}', 
                                        color=colors(layer), linestyle='-')
        plot_series_and_reference_on_ax(bias_gradient_norms_ax, layer_df['iterations'].tolist(), 
                                        layer_df['pre_activation_average_gradient_norm'].replace(0, numpy.nan).tolist(), f'|| E[∂L/∂u] || {layer}', 
                                        color=colors(layer), linestyle='--')

    plot_architecture_changes_to_ax(weights_gradient_norms_ax, sample_size, batch_size, model_metrics)
    plot_architecture_changes_to_ax(weights_gradient_norms_ax, sample_size, batch_size, model_metrics)
    for ax in (weights_gradient_norms_ax, bias_gradient_norms_ax): ax.legend(); ax.set_yscale('log')
    if canvas: draw_figure_into_canvas(fig, canvas)
    if save_figure_path: save_figure(fig, {'batch_size': batch_size, 'sample_size': sample_size, **kwargs}, save_figure_path, figure_name_parameters, 'weights_norm_and_biases')



def plot_weights_and_biases_gradient_norms(model_metrics, batch_size, sample_size, canvas=None, save_figure_path=None, figure_name_parameters=None, *args, **kwargs):
    fig, (weights_gradient_norms_ax, bias_gradient_norms_ax) = matplotlib.pyplot.subplots(1, 2, figsize=(16, 8))
    weights_gradient_norms_ax.set_title('Weights gradient norms')
    weights_gradient_norms_ax.set_xlabel('iteration')
    weights_gradient_norms_ax.set_ylabel('L2 norm')
    bias_gradient_norms_ax.set_title('Bias gradient norms')
    bias_gradient_norms_ax.set_xlabel('iteration')
    bias_gradient_norms_ax.set_ylabel('L2 norm')

    model_metrics_df = pandas.DataFrame(model_metrics)
    model_metrics_df['iterations'] = model_metrics_df['epoch'] * sample_size / batch_size
    colors = matplotlib.pyplot.cm.get_cmap('tab10', max(model_metrics_df['layer'].max(), 10))
    model_metrics_df['most_unstable_neuron'] = model_metrics_df.apply(
        lambda row: (numpy.array(row['positive_margins_count']) / numpy.array(row['active_samples_count'])).argmin(axis=0), axis='columns'
    )
    metrics_to_plot = ['gradients_average_norm', 'average_gradient_norm', 'pre_activation_gradient_average_norm', 'pre_activation_average_gradient_norm']
    for metric in metrics_to_plot:
        model_metrics_df[metric] = model_metrics_df.apply(lambda row: row[metric][row['most_unstable_neuron']] if row[metric] is not numpy.nan else numpy.nan, axis='columns')

    for layer, layer_df in model_metrics_df.groupby('layer'):
        layer_df = layer_df.dropna(subset=metrics_to_plot)
        if layer != model_metrics_df['layer'].max():
            plot_series_and_reference_on_ax(weights_gradient_norms_ax, layer_df['iterations'].tolist(), 
                                            layer_df['gradients_average_norm'].replace(0, numpy.nan).tolist(), f'$E[|| ∂L/∂w^{layer} ||]$', 
                                            color=colors(layer), linestyle='-')
            plot_series_and_reference_on_ax(weights_gradient_norms_ax, layer_df['iterations'].tolist(), 
                                            layer_df['average_gradient_norm'].replace(0, numpy.nan).tolist(), f'$|| E[∂L/∂w^{layer}] ||$', 
                                            color=colors(layer), linestyle='--')
            plot_series_and_reference_on_ax(bias_gradient_norms_ax, layer_df['iterations'].tolist(), 
                                            layer_df['pre_activation_gradient_average_norm'].replace(0, numpy.nan).tolist(), f'$E[|| ∂L/∂u^{layer} ||]$', 
                                            color=colors(layer), linestyle='-')
            plot_series_and_reference_on_ax(bias_gradient_norms_ax, layer_df['iterations'].tolist(), 
                                            layer_df['pre_activation_average_gradient_norm'].replace(0, numpy.nan).tolist(), f'$|| E[∂L/∂u^{layer}] ||$', 
                                            color=colors(layer), linestyle='--')

    for ax in (weights_gradient_norms_ax, bias_gradient_norms_ax): ax.legend(); ax.set_yscale('log')
    if canvas: draw_figure_into_canvas(fig, canvas)
    if save_figure_path: save_figure(fig, {'batch_size': batch_size, 'sample_size': sample_size, **kwargs}, save_figure_path, figure_name_parameters, 'train_loss_and_accuracy')

def plot_weights_norm_and_biases(model_metrics, batch_size, sample_size, canvas=None, bias=None, save_figure_path=None, figure_name_parameters=None, *args, **kwargs):
    fig, ax = matplotlib.pyplot.subplots(figsize=(10, 10))
    ax.set_xlabel('iteration')
    ax.set_ylabel('L2 norm')
    model_metrics_df = pandas.DataFrame(model_metrics)
    model_metrics_df['iterations'] = model_metrics_df['epoch'] * sample_size / batch_size
    colors = matplotlib.pyplot.cm.get_cmap('tab10', max(int(model_metrics_df['layer'].max() * model_metrics_df['hidden_units'].max()), 10))
    color_index = 0
    for (layer, hidden_units), layer_df in model_metrics_df.groupby(['layer', 'hidden_units']):
        for unit in range(int(hidden_units)):
            if layer != model_metrics_df['layer'].max():
                plot_series_and_reference_on_ax(ax, layer_df['iterations'].tolist(), layer_df['neurons_weights_norm'].apply(lambda units_norms: units_norms[unit]).tolist(), 
                                                label=f'$||w^{layer}_{unit}||$', color=colors(color_index), linestyle='-')
                if bias:
                    plot_series_and_reference_on_ax(ax, layer_df['iterations'].tolist(), layer_df['biases'].apply(lambda biases: biases[unit]).tolist(), 
                                                    label=f'$b^{layer}_{unit}$', color=colors(color_index), linestyle=':')
                    plot_series_and_reference_on_ax(ax, layer_df['iterations'].tolist(), (layer_df['neurons_weights_norm'].apply(lambda units_norms: units_norms[unit]) + 
                                                                                        layer_df['biases'].apply(lambda biases: biases[unit])).tolist(), 
                                                    label=f'$||w^{layer}_{unit}|| + b^{layer}_{unit}$', color=colors(color_index), linestyle='--')
                
            else:
                plot_series_and_reference_on_ax(ax, layer_df['iterations'].tolist(), layer_df['neurons_weights_norm'].apply(lambda units_norms: units_norms[unit]).tolist(), 
                                                label=f'$||v_{unit}||$', color=colors(color_index), linestyle='-')
                
            color_index += 1

    ax.legend()
    if canvas: draw_figure_into_canvas(fig, canvas)
    if save_figure_path: save_figure(fig, {'batch_size': batch_size, 'sample_size': sample_size, **kwargs}, save_figure_path, figure_name_parameters, 'train_loss_and_accuracy')