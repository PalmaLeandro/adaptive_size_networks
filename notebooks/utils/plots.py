import io, numpy, pandas, torch, matplotlib.pyplot, matplotlib.cm, matplotlib.colors, matplotlib.patches, ipywidgets
from .persistance import file_path_from_parameters, check_path

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

def limit_series(series, upper_lim=None, lower_lim=None):
    if upper_lim is not None:
        series = numpy.where(series >= upper_lim, numpy.nan, series)

    if lower_lim is not None:
        series = numpy.where(series <= lower_lim, numpy.nan, series)

    return series

def plot_series_and_reference_on_ax(ax, x, y, label, color=None, linestyle=None, fill_between_y=None, 
                                    lower_bound=None, upper_bound=None, log_scale=False, 
                                    upper_lim=None, lower_lim=None, axis_position='left'):
    if log_scale: lower_lim = 0. if lower_lim is None else max(0., lower_lim)

    y = limit_series(y, upper_lim, lower_lim)
    if lower_bound is not None: lower_bound = limit_series(lower_bound, upper_lim, lower_lim)
    if upper_bound is not None: upper_bound = limit_series(upper_bound, upper_lim, lower_lim)

    [line] = ax.plot(x, y, c=color, label=label, linestyle=linestyle)
    x_lower_lim, x_upper_lim = ax.get_xlim()
    ax.hlines(y[-1], x_lower_lim, x_upper_lim, color='gray', alpha=.3, linestyle='--')
    if axis_position == 'left':
        ax.text(x_lower_lim, y[-1], f'{y[-1]:.5f}', ha='left')
    elif axis_position == 'right':
        ax.text(x_upper_lim, y[-1], f'{y[-1]:.5f}', ha='left')

    if fill_between_y is not None: 
        up = limit_series(y + fill_between_y, upper_lim, lower_lim)
        down = limit_series(y - fill_between_y, upper_lim, lower_lim)
        if log_scale:
            up = numpy.where(up <= 0, numpy.nan, up)
            down = numpy.where(down <= 0, numpy.nan, down)

        ax.fill_between(x, up, down, color=color, alpha=0.1)

    if lower_bound is not None: ax.plot(x, lower_bound, linestyle=':', c=line.get_color())
    if upper_bound is not None: ax.plot(x, upper_bound, linestyle=':', c=line.get_color())
    if lower_lim or upper_lim: ax.set_ylim(lower_lim, upper_lim)
    if log_scale: ax.set_yscale('log')
    ax.set_xlim(x_lower_lim, x_upper_lim)

def plot_train_loss(ax, train_loss, test_loss, sample_size, epoch=0, batch_size=None, legend_loc='upper right', 
                    train_loss_color='blue', test_loss_color='red', prefix='', *args, **kwargs):
    ax.clear()
    ax.set_title(f'Train Loss (Epoch = {epoch})')
    ax.set_xlabel(('S' if batch_size < sample_size else '') + 'GD steps')
    batch_size = sample_size if batch_size is None else batch_size
    iterations = [iteration * sample_size / batch_size for iteration in range(epoch + 1)]
    plot_series_and_reference_on_ax(ax, iterations, train_loss, prefix + 'train', train_loss_color)
    plot_series_and_reference_on_ax(ax, iterations, test_loss, prefix + 'test', test_loss_color)
    if legend_loc:
        ax.legend(prop={'size': 16}, loc=legend_loc)

    ax.set_yscale('log')

def plot_samples_activation_hyperplanes(ax, dataloader, model, rotation_matrix, activation_resolution=1000, epoch=1, 
                                        mean=[0., 0.], scale=None, *args, **kwargs):
    ax.clear()
    mean = numpy.array(mean)
    inputs = []; labels = []
    for batch_inputs, batch_labels in dataloader: inputs.append(batch_inputs); labels.append(batch_labels)
    inputs, labels = torch.concatenate(inputs), torch.concatenate(labels)

    input_dimension = inputs.shape[-1]
    if input_dimension == 2:
        inputs_ = inputs if rotation_matrix is None else numpy.matmul(inputs, rotation_matrix.transpose())
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
            for neuron_index, (hyperplane_ys, neuron_weights, neuron_output_layer_sign) in enumerate(zip(neurons_ys, neurons_weights, neurons_output_layer_sign)):
                if neuron_index not in model.dead_units[0]:
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
                extent=[mean[0] - scale, mean[0] + scale, mean[1] - scale, mean[1] + scale], zorder=0, alpha=0.1)

def plot_samples_and_neurons(ax, dataloader, model, rotation_matrix, activation_resolution=1000, epoch=1,
                             mean=[0., 0.], scale=None, discard_dead_units=False, label_neurons=False, label_data=False, 
                             plot_activations=True, plot_activation_class=None, filter_classes=None, classes=2, *args, **kwargs):
    ax.clear()
    mean = numpy.array(mean)
    inputs = []; labels = []
    for batch_inputs, batch_labels in dataloader: inputs.append(batch_inputs); labels.append(batch_labels)
    inputs, labels = torch.concatenate(inputs), torch.concatenate(labels)
    input_dimension = inputs.shape[-1]

    inputs_ = inputs if rotation_matrix is None else numpy.matmul(inputs, rotation_matrix.transpose())
    ax.set_title(f'Input domain (Epoch = {epoch})')
    ax_lim = max(inputs_[:, 0].abs().max().item() * 1.1, inputs_[:, 1].abs().max().item() * 1.1)
    ax.hlines(0, ax_lim, ax_lim, color='black', zorder=1)
    ax.vlines(0, ax_lim, ax_lim, color='black', zorder=1)
    #for node in nodes:
    #    ax.add_patch(matplotlib.patches.Circle(node, experiment['epsilon'], color='k', alpha=.05))
    samples_classes = torch.argmax(labels, dim=1) if classes > 2 else labels
    colors = matplotlib.pyplot.get_cmap('RdYlBu', len(torch.unique(samples_classes)))(samples_classes)
    if filter_classes is not None:
        # Setting trasparency (alpha)
        colors[:, -1] = torch.isin(samples_classes, torch.tensor(filter_classes)).detach().cpu().numpy()

    ax.scatter(inputs_[:, 0], inputs_[:, 1], c=colors, label=samples_classes, zorder=2)
    if label_data:
        for sample_input_index, sample_input in enumerate(inputs_):
            ax.text(sample_input[0] * 1.05, sample_input[1] * 0.95, f'({sample_input[0]:.2e}, {sample_input[1]:.2e})', fontsize='large', c='k', zorder=3)

    input_layer = (model.input_layer.weight * model.output_layer.weight.norm(dim=0).unsqueeze(1)).detach().cpu().numpy()
    neurons_weights = numpy.matmul(input_layer, rotation_matrix.transpose())
    neurons_signs = model.neurons_sign
    l_1inf_norm = numpy.abs(neurons_weights).max(axis=1)
    neurons_weights[l_1inf_norm > ax_lim] *= (ax_lim / l_1inf_norm[l_1inf_norm > ax_lim, numpy.newaxis])

    if scale is not None:
        ax.set_xlim((mean[0] - scale) * 1.1, (mean[0] + scale) * 1.1); ax.set_ylim((mean[1] - scale) * 1.1, (mean[1] + scale) * 1.1)
    else:
        scale = ax_lim
    
    if len(neurons_weights):
        for neuron_index, (neuron_weights, neuron_sign) in enumerate(zip(neurons_weights, neurons_signs)):
            if not discard_dead_units or neuron_index not in model.dead_units[0]:
                neuron_weights = (neuron_weights[0], neuron_weights[1])
                norm = numpy.linalg.norm(neuron_weights)
                neuron_weights = list(map(lambda x: min(max(x, - ax_lim), ax_lim), neuron_weights))
                color = 'blue' if neuron_sign > 0 else ('red' if neuron_sign < 0 else 'black')
                ax.scatter(neuron_weights[0], neuron_weights[1], marker='+', s=200, c=color, zorder=3)

                normalized_neuron_weights = numpy.array(neuron_weights) / norm
                ax.plot([0, normalized_neuron_weights[0]], [0, normalized_neuron_weights[1]], c=color, alpha=0.1)
                
                if label_neurons:
                    ax.text(neuron_weights[0] * 1.05, neuron_weights[1] * 0.95, str(neuron_index), fontsize='large', c=color, zorder=3)
                    
    if plot_activations and classes <= 2:
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
        activations = model(torch.Tensor(domain_mesh).to(model.device))
        if plot_activation_class is not None:
            activations = activations[:, plot_activation_class].cpu().detach().numpy()
        else:
            activations = ((torch.sign(activations) + 1.) * 0.5).cpu().detach().numpy()

        ax.imshow(numpy.reshape(activations, (activation_resolution, activation_resolution)),
                origin='lower', cmap='gray', vmin=0, vmax=1,
                extent=[mean[0] - scale, mean[0] + scale, mean[1] - scale, mean[1] + scale], zorder=0)

def plot_weights_gradient_norms(ax, model, model_metrics, batch_size, sample_size, epoch=0, discard_dead_units=False, *args, **kwargs):
    ax.clear()
    ax.set_title(f'Weights gradient norms (Epoch = {epoch})')
    ax.set_xlabel('iteration')
    ax.set_ylabel('L2 norm')
    positive_neurons = numpy.argwhere(model.output_layer.weight.squeeze(dim=0).detach().cpu().numpy() > 0).reshape(-1).tolist()
    if discard_dead_units:
        positive_neurons = [neuron_index for neuron_index in positive_neurons if neuron_index not in model.dead_units[0]]
    negative_neurons = numpy.argwhere(model.output_layer.weight.squeeze(dim=0).detach().cpu().numpy() < 0).reshape(-1).tolist()
    if discard_dead_units:
        negative_neurons = [neuron_index for neuron_index in negative_neurons if neuron_index not in model.dead_units[0]]
    model_metrics_df = pandas.DataFrame(model_metrics)
    model_metrics_df['iterations'] = model_metrics_df['epoch'] * sample_size / batch_size
    blue = numpy.array(matplotlib.colors.to_rgba('blue', alpha=None))
    red = numpy.array(matplotlib.colors.to_rgba('red', alpha=None))
    metrics_to_plot = ['gradients_average_norm', 'average_gradient_norm', 'pre_activation_gradient_average_norm', 'pre_activation_average_gradient_norm']
    for metric in metrics_to_plot:
        model_metrics_df[f'{metric}_pos'] = model_metrics_df.apply(lambda row: numpy.average(numpy.array(row[metric])[positive_neurons]) if row[metric] is not numpy.nan else numpy.nan, axis='columns')
        model_metrics_df[f'{metric}_neg'] = model_metrics_df.apply(lambda row: numpy.average(numpy.array(row[metric])[negative_neurons]) if row[metric] is not numpy.nan else numpy.nan, axis='columns')

    for layer, layer_df in model_metrics_df.groupby('layer'):
        layer_df = layer_df.dropna(subset=metrics_to_plot)
        if layer != model_metrics_df['layer'].max():
            plot_series_and_reference_on_ax(ax, layer_df['iterations'].tolist(), layer_df['gradients_average_norm_pos'].replace(0, numpy.nan).tolist(), f'$E[|| ∂L/∂w_+^{layer} ||]$', 
                                            color=blue, linestyle='-')
            plot_series_and_reference_on_ax(ax, layer_df['iterations'].tolist(), layer_df['average_gradient_norm_pos'].replace(0, numpy.nan).tolist(), f'$|| E[∂L/∂w_+^{layer}] ||$', 
                                            color=blue, linestyle='--')
            plot_series_and_reference_on_ax(ax, layer_df['iterations'].tolist(), layer_df['gradients_average_norm_neg'].replace(0, numpy.nan).tolist(), f'$E[|| ∂L/∂w_-^{layer} ||]$', 
                                            color=red, linestyle='-')
            plot_series_and_reference_on_ax(ax, layer_df['iterations'].tolist(), layer_df['average_gradient_norm_neg'].replace(0, numpy.nan).tolist(), f'$|| E[∂L/∂w_-^{layer}] ||$', 
                                            color=red, linestyle='--')

    ax.legend()
    ax.set_yscale('log')

def plot_weights_norms(ax, model, model_metrics, batch_size, sample_size, epoch=None, discard_dead_units=False, *args, **kwargs):
    ax.clear()
    ax.set_title(f'Parameters norms (Epoch = {epoch})')
    ax.set_xlabel('iteration')
    ax.set_ylabel('L2 norm')
    neuron_indices = list(range(model.architecture[0])) 
    if discard_dead_units:
        neuron_indices = [index for index in neuron_indices if index not in model.dead_units[0]]

    model_metrics_df = pandas.DataFrame(model_metrics)
    model_metrics_df['iterations'] = model_metrics_df['epoch'] * sample_size / batch_size
    blue = matplotlib.colors.to_rgba('blue', alpha=None)
    red = matplotlib.colors.to_rgba('red', alpha=None)
    for layer, layer_df in model_metrics_df.groupby('layer'):
        layer_df = layer_df.dropna(subset=['neurons_weights_norm'])
        plot_series_and_reference_on_ax(ax, layer_df['iterations'].tolist(), 
                                        layer_df['neurons_weights_norm'].apply(lambda units_norms: numpy.average(numpy.array(units_norms)[neuron_indices])).tolist(), 
                                        label=f'$||w_+^{layer}||$', color=blue, linestyle='-')
        ratio_average = model_metrics_df.apply(lambda row: numpy.average(row['loss_gradient_inner_product_to_norm_ratio']), axis='columns').to_numpy()
        ratio_stddev = model_metrics_df.apply(lambda row: numpy.std(row['loss_gradient_inner_product_to_norm_ratio']), axis='columns').to_numpy()
        ratio_min = model_metrics_df.apply(lambda row: numpy.min(row['loss_gradient_inner_product_to_norm_ratio']), axis='columns').to_numpy()
        ratio_max = model_metrics_df.apply(lambda row: numpy.max(row['loss_gradient_inner_product_to_norm_ratio']), axis='columns').to_numpy()
        plot_series_and_reference_on_ax(ax, model_metrics_df['iterations'].tolist(), ratio_average, 
                                        fill_between_y=ratio_stddev, lower_bound=ratio_min, upper_bound=ratio_max,
                                        label=('$\lambda \,$' if regularization else '') + f'$E |<∂l(f(X)), v_j \phi(w_j^t X)>| \, / \, |v_j|||w_j||$' + ('$\lambda \,$' if regularization else ''))

    ax.legend()
    ax.set_yscale('log')

def plot_norms_min_max_variation(ax, initialization_scale, learning_rate, epoch, *args, **kwargs):
    iterations = [iteration for iteration in range(0, epoch + 1)]
    max_variation = initialization_scale * numpy.exp(learning_rate * numpy.array(iterations) / 4.)
    min_variation = initialization_scale * numpy.exp(learning_rate * numpy.array(iterations) / 4.)
    ax.plot(iterations, max_variation, linestyle='--', c='k', alpha=0.1, label='$max ∂||w||/∂t$')
    ax.plot(iterations, min_variation, linestyle='--', c='k', alpha=0.1, label='$min ∂||w||/∂t$')
