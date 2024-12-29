import io, os, sys, numpy, pandas, torch, matplotlib.pyplot, matplotlib.cm, ipywidgets, imageio
from .persistance import file_path_from_parameters, check_path

sys.path.append(os.path.abspath(os.path.join('..')))  # Allow repository modules to be imported 
HISTOGRAM_BINS = 50

def draw_figure_into_canvas(figure, canvas, *args, **kwargs):
    buffer = io.BytesIO()
    figure.tight_layout()
    figure.savefig(buffer, format='png', bbox_inches='tight')
    matplotlib.pyplot.close(figure)
    canvas.draw_image(ipywidgets.Image(value=buffer.getvalue()), width=canvas.width, height=canvas.height)

def save_figure(figure, name_parameters, save_figure_path='./plots/', figure_name='', *args, **parameters):
    check_path(f'{save_figure_path}{figure_name}')
    file_path = file_path_from_parameters(parameters=parameters, name_parameters=name_parameters, prefix=f'{save_figure_path}{figure_name}/', suffix='.png')
    figure.tight_layout()
    figure.savefig(file_path, format='png', bbox_inches='tight')

def create_indexing(*indices):
    return tuple(slice(None, idx) for idx in indices)

def save_figures_as_animation(figures_path, animation_name='animation', figures_indices=None, duration=5):
    figures = sorted([figure_file for figure_file in os.listdir(figures_path) if figure_file.endswith('.png')])
    if figures_indices is None:
        frames = [imageio.imread(os.path.join(figures_path, figure)) for figure in figures]
    else:
        frames = [imageio.imread(os.path.join(figures_path, figures[i])) for i in figures_indices]

    indexing = create_indexing(*[min([frame.shape[dimension] for frame in frames]) for dimension in range(len(frames[0].shape))])
    imageio.mimsave(os.path.join(figures_path, f'{animation_name}.gif'), [frame[indexing] for frame in frames], duration=5)

def plot_series_and_reference_on_ax(ax, x, y, label, color=None, linestyle=None):
    ax.plot(x, y, c=color, label=label, linestyle=linestyle)
    ax.hlines(y[-1], 0, x[-1], color='gray', alpha=.3, linestyle='--')
    ax.text(0, y[-1], f'{y[-1]:.5f}', ha='left')

def plot_architecture_changes_to_ax(ax, sample_size, batch_size, model_metrics):
    model_metrics_df = pandas.DataFrame(model_metrics)
    if 'architecture_change' in model_metrics_df:
        model_metrics_df['iterations'] = model_metrics_df['epoch'] * sample_size / batch_size
        colors = {
            'units_removed': 'orange',
            'layer_removed': 'red',
            'units_added': 'green',
            'layer_added': 'blue'
        }
        y_lims = ax.get_ylim()
        for architecture_change, architecture_change_df in model_metrics_df.groupby('architecture_change'):
            iterations = architecture_change_df['iterations'].unique()
            ax.vlines(iterations, [y_lims[0],] * len(iterations), [y_lims[1],] * len(iterations), 
                      color=colors[architecture_change], alpha=0.3, label=architecture_change)

def plot_train_loss_and_accuracy(train_loss, test_loss, train_accuracy, test_accuracy, sample_size, model_metrics, epoch=None,
                                 batch_size=None, canvas=None, save_figure_path=None, figure_name_parameters=None, *args, **kwargs):
    if canvas: matplotlib.pyplot.ioff()
    fig, (loss_ax, accuracy_ax) = matplotlib.pyplot.subplots(1, 2, figsize=(16, 8))
    epoch = epoch if epoch is not None else len(train_loss)
    loss_ax.set_title(f'Train Loss (Epoch = {epoch})')
    loss_ax.set_xlabel('iterations')
    accuracy_ax.set_title(f'Accuracy (Epoch = {epoch})')
    accuracy_ax.set_xlabel('iterations')
    batch_size = sample_size if batch_size is None else batch_size
    iterations = [iteration * sample_size / batch_size for iteration in range(epoch + 1)]
    plot_series_and_reference_on_ax(loss_ax, iterations, train_loss, 'train', 'b')
    plot_series_and_reference_on_ax(loss_ax, iterations, test_loss, 'test', 'r')
    plot_series_and_reference_on_ax(accuracy_ax, iterations, train_accuracy, 'train', 'b')
    plot_series_and_reference_on_ax(accuracy_ax, iterations, test_accuracy, 'test', 'r')
    plot_architecture_changes_to_ax(loss_ax, sample_size, batch_size, model_metrics)
    plot_architecture_changes_to_ax(accuracy_ax, sample_size, batch_size, model_metrics)
    for ax in (loss_ax, accuracy_ax): ax.legend()
    loss_ax.set_yscale('log')
    if canvas: draw_figure_into_canvas(fig, canvas)
    if save_figure_path: save_figure(fig, {'epoch': epoch, 'batch_size': batch_size, 'sample_size': sample_size, **kwargs}, save_figure_path, figure_name_parameters, 'train_loss_and_accuracy')

def histogram_bars(histogram_frequencies, histogram_bins):
    histogram_bins = histogram_bins.detach().cpu().numpy()[:-1]
    histogram_bins_pace = histogram_bins[1] - histogram_bins[0]
    histogram_bins += histogram_bins_pace / 2.
    histogram_frequencies = histogram_frequencies.detach().cpu().numpy()
    return histogram_bins, histogram_frequencies / histogram_frequencies.sum(), histogram_bins_pace

def plot_samples_activation_hyperplanes(dataloader, model, rotation_matrix, activation_resolution=1000, epoch=1, 
                                        canvas=None, mean=[0., 0.], scale=None, save_figure_path=None, figure_name_parameters=None, *args, **kwargs):
    mean = numpy.array(mean)
    inputs = []; labels = []
    for batch_inputs, batch_labels in dataloader: inputs.append(batch_inputs); labels.append(batch_labels)
    inputs, labels = torch.concatenate(inputs), torch.concatenate(labels)

    input_dimension = inputs.shape[-1]
    inputs_ = inputs if rotation_matrix is None else numpy.matmul(inputs, rotation_matrix.transpose())

    fig, ax = matplotlib.pyplot.subplots(figsize=(10, 10))
    ax.set_title(f'Input domain (Epoch = {epoch})')
    ax.hlines(0, inputs_[:, 0].min() * 1.1, inputs_[:, 0].max() * 1.1, color='black', zorder=1)
    ax.vlines(0, inputs_[:, 0].min() * 1.1, inputs_[:, 0].max() * 1.1, color='black', zorder=1)
    ax.scatter(inputs_[:, 0], inputs_[:, 1], c=labels, label=labels, cmap=matplotlib.cm.get_cmap('RdYlBu'), zorder=2)

    input_layer = model.layers[0]
    neurons_weights = numpy.matmul(input_layer.weight.detach().cpu().numpy(), rotation_matrix.transpose())
    number_of_neurons = len(neurons_weights)
    neurons_bias = input_layer.bias.detach().cpu().numpy() if input_layer.bias is not None else numpy.repeat(0., number_of_neurons)
    neurons_output_layer_sign = model.output_layer.weight.sign().squeeze().detach().cpu().tolist()

    if scale is not None:
        ax.set_xlim((mean[0] - scale) * 1.1, (mean[0] + scale) * 1.1); ax.set_ylim((mean[1] - scale) * 1.1, (mean[1] + scale) * 1.1)
    else:
        scale = max(inputs_[:, 0].max(), inputs_[:, 1].max()) * 1.1
    
    xs = numpy.linspace(mean[0] - scale, mean[0] + scale)
    if len(neurons_weights):
        neurons_ys = (neurons_weights[:, 0][:, None] * xs + neurons_bias[:, None]) / (- neurons_weights[:, 1][:, None])
        for hyperplane_ys, neuron_weights, neuron_output_layer_sign in zip(neurons_ys, neurons_weights, neurons_output_layer_sign):
            color = 'b' if neuron_output_layer_sign > 0 else 'r'
            ax.plot(xs, hyperplane_ys, c=color)
            ax.arrow(xs[len(xs) // 2], hyperplane_ys[len(hyperplane_ys) // 2], neuron_weights[0], neuron_weights[1], width=0.01, color=color)

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
              origin='lower', cmap='gray', vmin=0, vmax=1,
              extent=[mean[0] - scale, mean[0] + scale, mean[1] - scale, mean[1] + scale], zorder=0)
    
    if canvas: draw_figure_into_canvas(fig, canvas)
    if save_figure_path: save_figure(fig, {'epoch':epoch, **kwargs}, save_figure_path, figure_name_parameters, 'samples_activation_hyperplanes')
    return fig

def plot_weights_and_biases_gradient_norms(model_metrics, batch_size, sample_size, canvas=None, save_figure_path=None, figure_name_parameters=None, epoch=0, *args, **kwargs):
    fig, (weights_gradient_norms_ax, bias_gradient_norms_ax) = matplotlib.pyplot.subplots(1, 2, figsize=(16, 8))
    weights_gradient_norms_ax.set_title(f'Weights gradient norms (Epoch = {epoch})')
    weights_gradient_norms_ax.set_xlabel('iteration')
    weights_gradient_norms_ax.set_ylabel('L2 norm')
    bias_gradient_norms_ax.set_title(f'Bias gradient norms (Epoch = {epoch})')
    bias_gradient_norms_ax.set_xlabel('iteration')
    bias_gradient_norms_ax.set_ylabel('L2 norm')

    model_metrics_df = pandas.DataFrame(model_metrics)
    model_metrics_df['iterations'] = model_metrics_df['epoch'] * sample_size / batch_size
    colors = matplotlib.pyplot.cm.get_cmap('tab10', max(model_metrics_df['layer'].max(), 10))
    metrics_to_plot = ['gradients_average_norm', 'average_gradient_norm', 'pre_activation_gradient_average_norm', 'pre_activation_average_gradient_norm']
    for metric in metrics_to_plot:
        model_metrics_df[metric] = model_metrics_df.apply(lambda row: numpy.average(row[metric]), axis='columns')

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
    if save_figure_path: save_figure(fig, {'epoch': epoch, 'batch_size': batch_size, 'sample_size': sample_size, **kwargs}, save_figure_path, figure_name_parameters, 'weights_and_biases_gradient_norms')

def plot_weights_norm_and_biases(model_metrics, batch_size, sample_size, canvas=None, bias=None, save_figure_path=None, figure_name_parameters=None, epoch=None, *args, **kwargs):
    fig, ax = matplotlib.pyplot.subplots(figsize=(10, 10))
    ax.set_title(f'Parameters norms (Epoch = {epoch})')
    ax.set_xlabel('iteration')
    ax.set_ylabel('L2 norm')
    model_metrics_df = pandas.DataFrame(model_metrics)
    model_metrics_df['iterations'] = model_metrics_df['epoch'] * sample_size / batch_size
    colors = matplotlib.pyplot.cm.get_cmap('tab10', max(int(model_metrics_df['layer'].max()), 10))
    color_index = 0
    for layer, layer_df in model_metrics_df.groupby('layer'):
        layer_df = layer_df.dropna(subset=['neurons_weights_norm'] + (['biases'] if bias else []))
        if layer != model_metrics_df['layer'].max():
            plot_series_and_reference_on_ax(ax, layer_df['iterations'].tolist(), layer_df['neurons_weights_norm'].apply(lambda units_norms: numpy.average(units_norms)).tolist(), 
                                            label=f'$||w^{layer}||$', color=colors(color_index), linestyle='-')
            if bias:
                plot_series_and_reference_on_ax(ax, layer_df['iterations'].tolist(), layer_df['biases'].apply(lambda biases: numpy.average(biases)).tolist(), 
                                                label=f'$b^{layer}$', color=colors(color_index), linestyle=':')
                plot_series_and_reference_on_ax(ax, layer_df['iterations'].tolist(), (layer_df['neurons_weights_norm'].apply(lambda units_norms: numpy.average(units_norms)) + 
                                                                                    layer_df['biases'].apply(lambda biases: numpy.average(biases))).tolist(), 
                                                label=f'$||w^{layer}|| + b^{layer}$', color=colors(color_index), linestyle='--')
            
        else:
            plot_series_and_reference_on_ax(ax, layer_df['iterations'].tolist(), layer_df['neurons_weights_norm'].apply(lambda units_norms: numpy.average(units_norms)).tolist(), 
                                            label=f'$||v||$', color=colors(color_index), linestyle='-')
            
        color_index += 1

    ax.legend()
    if canvas: draw_figure_into_canvas(fig, canvas)
    if save_figure_path: save_figure(fig, {'epoch': epoch, 'batch_size': batch_size, 'sample_size': sample_size, **kwargs}, save_figure_path, figure_name_parameters, 'weights_norm_and_biases')
