import io, functools, numpy, matplotlib.pyplot

MINIMIZATION_LOSSES = ['squared error', 'cross entropy']

def draw_figure_into_canvas(figure, canvas):
    import ipywidgets

    buffer = io.BytesIO()
    figure.tight_layout()
    figure.savefig(buffer, format='png', bbox_inches='tight')
    matplotlib.pyplot.close(figure)
    canvas.draw_image(ipywidgets.Image(value=buffer.getvalue()), width=canvas.width, height=canvas.height)

def plot_experiment(experiment, canvas=None, second_axis=None):
    sample_size, models_runs, batch_size, train_loss, test_loss, time_unit, legend = extract_variables(experiment)
    models_distinctions = sorted(list(set([model_run['distinction'] for model_run in models_runs])))
    if canvas: matplotlib.pyplot.ioff()

    fig, axis = matplotlib.pyplot.subplots(2, 2, figsize=(16, 16))
    ((iterations_train_ax, iterations_test_ax), (time_train_ax, time_test_ax)) = axis
    axis = [iterations_train_ax, iterations_test_ax, time_train_ax, time_test_ax]
    twin_axis = [ax.twinx() for ax in axis] if second_axis is not None else None
    
    train_y_lim_min, train_y_lim_max, test_y_lim_min, test_y_lim_max, max_epochs, max_train_time = 5, 0.1, 5, 0.1, 0, 0
    train_minimization, test_minimization = parse_optimization_goal(train_loss, test_loss)
    
    for models_distinction in models_distinctions:
        avg_train_loss, std_train_loss, avg_test_loss, std_test_loss, avg_train_time, epochs, iterations, second_ax_series = \
            get_series_to_plot(models_runs, models_distinction, sample_size, batch_size, second_axis)
        
        min_train, max_train = avg_train_loss.min(), avg_train_loss.max()
        min_test, max_test = avg_test_loss.min(), avg_test_loss.max()
        max_iterations = float(epochs * sample_size / batch_size) + 1   
        max_epochs, max_train_time = max(max_epochs, epochs), max(max_train_time, avg_train_time.max())
        train_y_lim_min, test_y_lim_min = min(train_y_lim_min, min_train), min(test_y_lim_min, min_test)
        train_y_lim_max, test_y_lim_max = max(train_y_lim_max, max_train), max(test_y_lim_max, max_test)

        plot_series(avg_train_loss, std_train_loss, avg_test_loss, std_test_loss, avg_train_time, models_distinction, iterations, *axis, 
                    second_ax_series, twin_axis)
        plot_hlines(train_minimization, test_minimization, max_iterations, min_train, max_train, min_test, max_test, max_train_time, *axis)
    
    max_iterations = max_epochs * sample_size / batch_size
    set_axis_legend(legend, *axis)
    set_axis_limits(max_iterations, max_train_time, train_y_lim_min, train_y_lim_max, test_y_lim_min, test_y_lim_max, *axis)
    set_axis_xticks(*axis)
    set_axis_scales(train_minimization, test_minimization, *axis)
    set_axis_titles(*axis)
    set_axis_xlabel(batch_size, time_unit, sample_size, *axis)
    set_axis_ylabel(train_loss, test_loss, *axis, second_axis, twin_axis)
    if canvas: draw_figure_into_canvas(fig, canvas)

def extract_variables(experiment):
    return (
        experiment['sample_size'], 
        experiment['models_runs'],
        experiment['batch_size'],
        experiment['train'],
        experiment['test'],
        experiment['train_time'],
        experiment.get('distinction', 'run')
    )

def parse_optimization_goal(train_loss, test_loss):
    train_minimization = test_minimization = False
    if any([loss for loss in MINIMIZATION_LOSSES if loss in train_loss.lower()]): 
        train_minimization = True

    if any([loss for loss in MINIMIZATION_LOSSES if loss in test_loss.lower()]): 
        test_minimization = True

    return train_minimization, test_minimization

def set_axis_titles(iterations_train_ax, iterations_test_ax, time_train_ax, time_test_ax):
    iterations_train_ax.set_title('Train loss'); iterations_test_ax.set_title('Test loss')
    time_train_ax.set_title('Train loss'); time_test_ax.set_title('Test loss')

def set_axis_xlabel(batch_size, time_unit, sample_size, iterations_train_ax, iterations_test_ax, time_train_ax, time_test_ax):
    x_axis_text = f'{"SGD (batch size = " + str(batch_size) + ")" if batch_size < sample_size else "GD"} iterations'
    iterations_train_ax.set_xlabel(x_axis_text); iterations_test_ax.set_xlabel(x_axis_text)
    x_axis_text = f'Train time ({time_unit})'
    time_train_ax.set_xlabel(x_axis_text); time_test_ax.set_xlabel(x_axis_text)

def set_axis_ylabel(train_loss, test_loss, iterations_train_ax, iterations_test_ax, time_train_ax, time_test_ax,
                    second_axis_label=None, second_axis=None):
    iterations_train_ax.set_ylabel(train_loss)
    iterations_test_ax.set_ylabel(test_loss)
    time_train_ax.set_ylabel(train_loss)
    time_test_ax.set_ylabel(test_loss)

    if second_axis_label is not None and second_axis is not None:
        iterations_train_twin_ax, iterations_test_twin_ax, time_train_twin_ax, time_test_twin_ax = second_axis
        iterations_train_twin_ax.set_ylabel(second_axis_label)
        iterations_test_twin_ax.set_ylabel(second_axis_label)
        time_train_twin_ax.set_ylabel(second_axis_label)
        time_test_twin_ax.set_ylabel(second_axis_label)

def set_axis_scales(train_minimization, test_minimization, iterations_train_ax, iterations_test_ax, time_train_ax, time_test_ax):
    #iterations_train_ax.set_xscale('log'); iterations_test_ax.set_xscale('log')
    if train_minimization: iterations_train_ax.set_yscale('log'); time_train_ax.set_yscale('log')
    if test_minimization: iterations_test_ax.set_yscale('log'); time_test_ax.set_yscale('log')

def get_series_to_plot(models_runs, models_distinction, sample_size, batch_size, second_axis=None):
    model_runs = [model_run for model_run in models_runs if model_run['distinction'] == models_distinction]
    train_losses = [model_run['train'] for model_run in model_runs]
    test_losses = [model_run['test'] for model_run in model_runs]
    train_times = [model_run['train_time'] for model_run in model_runs]
    epochs = max(*[len(model_run['train']) for model_run in model_runs], *[len(model_run['test']) for model_run in model_runs])

    train_losses = numpy.array([train_loss + ([] if len(train_loss) == epochs else ([numpy.nan] * (epochs - len(train_loss))))
                                for train_loss in train_losses])
    test_losses = numpy.array([test_loss + ([] if len(test_loss) == epochs else ([numpy.nan] * (epochs - len(test_loss))))
                                for test_loss in test_losses])
    train_times = numpy.array([train_time + ([] if len(train_time) == epochs else ([numpy.nan] * (epochs - len(train_time))))
                                for train_time in train_times])

    avg_train_loss, std_train_loss = numpy.nanmean(train_losses, axis=0), numpy.nanstd(train_losses, axis=0)
    avg_test_loss, std_test_loss = numpy.nanmean(test_losses, axis=0), numpy.nanstd(test_losses, axis=0)
    avg_train_time = numpy.nanmean(train_times, axis=0)

    iterations = [iteration * sample_size / batch_size for iteration in range(epochs)]
    
    if second_axis is None:
        return avg_train_loss, std_train_loss, avg_test_loss, std_test_loss, avg_train_time, epochs, iterations, None

    else:
        [second_ax_series] = [model_run[second_axis] for model_run in model_runs]
        return avg_train_loss, std_train_loss, avg_test_loss, std_test_loss, avg_train_time, epochs, iterations, second_ax_series

def plot_series(avg_train_loss, std_train_loss, avg_test_loss, std_test_loss, avg_train_time, models_distinction, iterations,
                iterations_train_ax, iterations_test_ax, time_train_ax, time_test_ax, second_ax_series=None, second_axis=None):
    iterations_train_ax.plot(iterations, avg_train_loss, label=models_distinction, zorder=3)
    iterations_train_ax.fill_between(iterations, avg_train_loss - std_train_loss, avg_train_loss + std_train_loss, zorder=2, alpha=0.2)
    iterations_test_ax.plot(iterations, avg_test_loss, label=models_distinction, zorder=3)
    iterations_test_ax.fill_between(iterations, avg_test_loss - std_test_loss, avg_test_loss + std_test_loss, zorder=2, alpha=0.2)

    time_train_ax.plot(avg_train_time, avg_train_loss, label=models_distinction, zorder=3)
    time_train_ax.fill_between(avg_train_time, avg_train_loss - std_train_loss, avg_train_loss + std_train_loss, zorder=2, alpha=0.2)
    time_test_ax.plot(avg_train_time, avg_test_loss, label=models_distinction, zorder=3)
    time_test_ax.fill_between(avg_train_time, avg_test_loss - std_test_loss, avg_test_loss + std_test_loss, zorder=2, alpha=0.2)

    if second_ax_series is not None and second_axis is not None:
        iterations_train_twin_ax, iterations_test_twin_ax, time_train_twin_ax, time_test_twin_ax = second_axis
        iterations_train_twin_ax.plot(iterations, second_ax_series, label=models_distinction, zorder=3, alpha=0.5, linestyle=':')
        iterations_test_twin_ax.plot(iterations, second_ax_series, label=models_distinction, zorder=3, alpha=0.5, linestyle=':')
        time_train_twin_ax.plot(avg_train_time, second_ax_series, label=models_distinction, zorder=3, alpha=0.5, linestyle=':')
        time_test_twin_ax.plot(avg_train_time, second_ax_series, label=models_distinction, zorder=3, alpha=0.5, linestyle=':')

def plot_hlines(train_minimization, test_minimization, max_iterations, min_train, max_train, min_test, max_test, 
                max_train_time, iterations_train_ax, iterations_test_ax, time_train_ax, time_test_ax):
    if train_minimization:
        iterations_train_ax.hlines(min_train, 0, max_iterations, colors='grey', linestyles='dashed')
        iterations_train_ax.text(0, min_train, f'{min_train:.2f}', horizontalalignment='right')
        time_train_ax.hlines(min_train, 0, max_train_time, colors='grey', linestyles='dashed')
        time_train_ax.text(0, min_train, f'{min_train:.2f}', horizontalalignment='right')
    
    else:
        iterations_train_ax.hlines(max_train, 0, max_iterations, colors='grey', linestyles='dashed')
        iterations_train_ax.text(0, max_train, f'{max_train:.2f}', horizontalalignment='right')
        time_train_ax.hlines(max_train, 0, max_train_time, colors='grey', linestyles='dashed')
        time_train_ax.text(0, max_train, f'{max_train:.2f}', horizontalalignment='right')

    if test_minimization:
        iterations_test_ax.hlines(min_test, 0, max_iterations, colors='grey', linestyles='dashed')
        iterations_test_ax.text(0, min_test, f'{min_test:.2f}', horizontalalignment='right')
        time_test_ax.hlines(min_test, 0, max_train_time, colors='grey', linestyles='dashed')
        time_test_ax.text(0, min_test, f'{min_test:.2f}', horizontalalignment='right')
    
    else:
        iterations_test_ax.hlines(max_test, 0, max_iterations, colors='grey', linestyles='dashed')
        iterations_test_ax.text(0, max_test, f'{max_test:.2f}', horizontalalignment='right')
        time_test_ax.hlines(max_test, 0, max_train_time, colors='grey', linestyles='dashed')
        time_test_ax.text(0, max_test, f'{max_test:.2f}', horizontalalignment='right')

def set_axis_legend(legend, iterations_train_ax, iterations_test_ax, time_train_ax, time_test_ax):
    iterations_train_ax.legend(title=legend)
    iterations_test_ax.legend(title=legend)
    time_train_ax.legend(title=legend)
    time_test_ax.legend(title=legend)

def set_axis_limits(max_iterations, max_train_time, train_y_lim_min, train_y_lim_max, test_y_lim_min, test_y_lim_max, 
                    iterations_train_ax, iterations_test_ax, time_train_ax, time_test_ax):
    iterations_train_ax.set_xlim(0, max_iterations)
    iterations_test_ax.set_xlim(0, max_iterations)
    iterations_train_ax.set_ylim(train_y_lim_min, train_y_lim_max)
    iterations_test_ax.set_ylim(test_y_lim_min, test_y_lim_max)
    
    time_train_ax.set_xlim(0, max_train_time)
    time_test_ax.set_xlim(0, max_train_time)
    time_train_ax.set_ylim(train_y_lim_min, train_y_lim_max)
    time_test_ax.set_ylim(test_y_lim_min, test_y_lim_max)

def set_axis_xticks(iterations_train_ax, iterations_test_ax, time_train_ax, time_test_ax):
    ticks_args = dict(style='sci', axis='x', scilimits=(0,0))
    
    iterations_train_ax.ticklabel_format(**ticks_args)
    iterations_test_ax.ticklabel_format(**ticks_args)
    iterations_train_ax.xaxis.major.formatter._useMathText = iterations_test_ax.xaxis.major.formatter._useMathText = True

    time_train_ax.ticklabel_format(**ticks_args)
    time_test_ax.ticklabel_format(**ticks_args)
    time_train_ax.xaxis.major.formatter._useMathText = time_test_ax.xaxis.major.formatter._useMathText = True

def plot_samples_and_model_activation(dataloader, model, rotation_matrix=None, input_dimension=None, activation_resolution=1000, epoch=None,
                                      epoch_frequency=None, canvas=None, mean=None, scale=None, **kwargs):
      if epoch is not None and epoch_frequency is not None and epoch % epoch_frequency != 0: return
      import torch

      _, (inputs, labels) = next(enumerate(dataloader))
      inputs_ = inputs if rotation_matrix is None else numpy.matmul(inputs, rotation_matrix.transpose())

      neurons_weights = model.input_layer.weight.detach().cpu().numpy()
      positive_neurons_weights = numpy.matmul(neurons_weights[:model.hidden_units // 2], rotation_matrix.transpose())
      negative_neurons_weights = numpy.matmul(neurons_weights[model.hidden_units // 2:], rotation_matrix.transpose())
      
      if scale is None:
        scale = max(inputs_[:, 0].max(), inputs_[:, 1].max(), 
                               positive_neurons_weights[:, 0].max(), positive_neurons_weights[:, 1].max(), 
                               negative_neurons_weights[:, 0].max(), negative_neurons_weights[:, 1].max())

      fig, ax = matplotlib.pyplot.subplots(figsize=(10, 10))

      domain_mesh = numpy.array([
            numpy.concatenate(dimension_repetition)
            for dimension_repetition
            in numpy.meshgrid(*([numpy.linspace(-scale, scale, activation_resolution)] * 2))
      ]).transpose()

      if mean is not None:
          domain_mesh += numpy.array(mean)
      
      if input_dimension > 2:
        domain_mesh = numpy.concatenate([domain_mesh, numpy.repeat(numpy.repeat(0., input_dimension - 2)[numpy.newaxis, :], len(domain_mesh), axis=0)], axis=1)
      
      domain_mesh = numpy.matmul(domain_mesh, rotation_matrix)
      domain_mesh = numpy.concatenate([domain_mesh, numpy.repeat(0., len(domain_mesh), axis=0).reshape(-1, 1)], axis=1)
      activations = ((torch.sign(model(torch.Tensor(domain_mesh).to(model.device))) + 1.) * 0.5).cpu().detach().numpy()
      ax.imshow(numpy.reshape(activations, (activation_resolution, activation_resolution)),
                origin='lower', cmap='gray', vmin=0, vmax=1, zorder=0,
                extent=[-scale, scale, -scale, scale])

      
      ax.scatter(inputs_[:, 0], inputs_[:, 1], c=labels)
      ax.scatter(positive_neurons_weights[:, 0], positive_neurons_weights[:, 1], c='b', marker='+', s=500)
      ax.scatter(negative_neurons_weights[:, 0], negative_neurons_weights[:, 1], c='r', marker='_', s=500)
      
      if canvas: draw_figure_into_canvas(fig, canvas)
      
      return fig
      
def plot_samples_and_model_activation_on_canvas(canvas):
    return functools.partial(plot_samples_and_model_activation, canvas=canvas)