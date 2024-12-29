import sys, os, numpy, torch, matplotlib.pyplot, pandas

sys.path.append(os.path.abspath(os.path.join('..')))  # Allow repository modules to be imported

from utils.plots import histogram_bars, plot_series_and_reference_on_ax, draw_figure_into_canvas, save_figure

HISTOGRAM_BINS = 50

def plot_model_metrics(model_metrics, batch_size, sample_size, target_accuracy, canvas=None, save_figure_path=None, figure_name_parameters=None, epoch=0, *args, **kwargs):
    fig, (margins_histogram_ax, positive_margins_ax, gradient_norms_ax) = matplotlib.pyplot.subplots(1, 3, figsize=(18, 6))
    margins_histogram_ax.set_title(f'Margins (Epoch = {epoch})')
    margins_histogram_ax.set_xlabel('margin (< ∂L/∂w, w > / || w || )')
    margins_histogram_ax.set_ylabel('% samples')
    positive_margins_ax.set_title(f'Growth metrics (Epoch = {epoch})')
    positive_margins_ax.set_xlabel('iteration')
    positive_margins_ax.set_ylabel('samples')
    gradient_norms_ax.set_title(f'Gradient norms (Epoch = {epoch})')
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
    if save_figure_path: save_figure(fig, {'epoch': epoch, 'batch_size': batch_size, 'sample_size': sample_size, **kwargs}, save_figure_path, figure_name_parameters, 'model_metrics')
    