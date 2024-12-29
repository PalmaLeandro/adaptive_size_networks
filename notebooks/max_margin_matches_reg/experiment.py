import sys, os, time, pandas, torch

sys.path.append(os.path.abspath(os.path.join('..')))  # Allow repository modules to be imported

from utils.optimization import initialize, train, test, accuracy, get_random_states, set_random_states
from utils.persistance import experiment_exists, load_experiment, save_experiment
from utils.plots import plot_series_and_reference_on_ax

def model_summary(model, predictions, labels, model_metrics, epoch, batch_size, regularization=0., *args, **kwargs):
    labels = ((labels * 2.) - 1.)
    margins = predictions * labels
    min_margin = margins.min().item()
    min_margin = min_margin if min_margin > 0 else None
    exp_margins = (- margins).exp()
    del margins
    
    exp_margins_sum = exp_margins.sum().item()
    neurons_outputs = model.output_layer.weight * model.activations[-1]
    inner_products = (neurons_outputs.T @ (exp_margins * labels).unsqueeze(1)).squeeze() / batch_size 
    del exp_margins, neurons_outputs
    
    norms = model.norms
    non_zero_neurons = norms.nonzero().flatten().detach().cpu().tolist()

    norms = norms[non_zero_neurons]
    loss_gradient_inner_product_to_norm_ratio = inner_products[non_zero_neurons] / norms
    #zeroed_out_neurons = torch.zeros(initial_hidden_units - len(non_zero_neurons))
    #norms = torch.cat([norms, zeroed_out_neurons])
    #loss_gradient_inner_product_to_norm_ratio = torch.cat([loss_gradient_inner_product_to_norm_ratio, zeroed_out_neurons])

    del inner_products
    loss_gradient_inner_product_to_norm_ratio_avg = loss_gradient_inner_product_to_norm_ratio.mean().item()
    loss_gradient_inner_product_to_norm_ratio_std = loss_gradient_inner_product_to_norm_ratio.std().item()
    loss_gradient_inner_product_to_norm_ratio_min = loss_gradient_inner_product_to_norm_ratio.min().item()
    loss_gradient_inner_product_to_norm_ratio_max = loss_gradient_inner_product_to_norm_ratio.max().item()
    del loss_gradient_inner_product_to_norm_ratio

    norms_avg = norms.mean().item()
    norms_std = norms.std().item()
    norms_min = norms.min().item()
    norms_max = norms.max().item()
    norm = norms.sum().item()
    del norms

    Lambda = exp_margins_sum / (norm * min_margin ** (1. - (1. / 2.))) if min_margin else None
    
    input_weights = model.input_layer.weight
    if input_weights.grad is not None:
        identity = torch.eye(input_weights.shape[1])
        input_weights_norm = input_weights.norm(dim=1).reshape(-1, 1, 1)
        projection_matrices = identity - ((input_weights.unsqueeze(2) @ input_weights.unsqueeze(1)) / input_weights_norm ** 2)
        changes_in_directions = (projection_matrices @ input_weights.grad.unsqueeze(2) / input_weights_norm ** 2).norm(dim=1).sum().item()

        del identity, input_weights_norm, projection_matrices
    else:
        changes_in_directions = 0

    unique_neuron_angles = len(torch.atan2(model.input_layer.weight[:, 1], model.input_layer.weight[:, 0]).round(decimals=3).unique())

    metrics_to_add = {'epoch': epoch, 'min_margin': min_margin, 'exp_margins_sum': exp_margins_sum, 'norm': norm, 
        'loss_gradient_inner_product_to_norm_ratio_avg': loss_gradient_inner_product_to_norm_ratio_avg,
        'loss_gradient_inner_product_to_norm_ratio_std': loss_gradient_inner_product_to_norm_ratio_std,
        'loss_gradient_inner_product_to_norm_ratio_min': loss_gradient_inner_product_to_norm_ratio_min,
        'loss_gradient_inner_product_to_norm_ratio_max': loss_gradient_inner_product_to_norm_ratio_max,
        'norms_avg': norms_avg, 'norms_std': norms_std,
        'norms_min': norms_min, 'norms_max': norms_max,
        'changes_in_directions': changes_in_directions,
        'non_zero_neurons': len(non_zero_neurons),
        'unique_neuron_angles': unique_neuron_angles
    }
    if Lambda: metrics_to_add.update({'Lambda': Lambda})
    if regularization: metrics_to_add.update({'regularization': regularization})
    model_metrics.append(metrics_to_add)

def plot_weights_norms(ax, model_metrics, epoch, sample_size, batch_size, *args, **kwargs):
    ax.clear()
    ax.set_title(f'Neurons norms (Epoch = {epoch})')
    ax.set_xlabel(('S' if batch_size < sample_size else '') + 'GD steps')
    model_metrics_df = pandas.DataFrame(model_metrics)
    model_metrics_df['iterations'] = model_metrics_df['epoch'] * sample_size / batch_size
    model_metrics_df = model_metrics_df[model_metrics_df['norms_avg'].notna()]
    plot_series_and_reference_on_ax(ax, model_metrics_df['iterations'].values, model_metrics_df['norms_avg'].values, 
                                        fill_between_y=model_metrics_df['norms_std'].values, 
                                        lower_bound=model_metrics_df['norms_min'].values, 
                                        upper_bound=model_metrics_df['norms_max'].values,
                                        label='$\|v_j\|_2\|w_j\|_2$')

    ax.legend(prop={'size': 16}, loc='upper right')

def plot_non_zero_norms(ax, model_metrics, sample_size, batch_size, *args, **kwargs):
    ax.clear()
    model_metrics_df = pandas.DataFrame(model_metrics)
    model_metrics_df['iterations'] = model_metrics_df['epoch'] * sample_size / batch_size
    model_metrics_df = model_metrics_df[model_metrics_df['non_zero_neurons'].notna()]
    ax.plot(model_metrics_df['iterations'].values, model_metrics_df['non_zero_neurons'].values, 
            label='$\|\\alpha\|_0$', c='r', linestyle='--')

    ax.legend(prop={'size': 16}, loc='upper left')

def plot_changes_in_directions(ax, model_metrics, sample_size, batch_size, *args, **kwargs):
    ax.clear()
    model_metrics_df = pandas.DataFrame(model_metrics)
    model_metrics_df['iterations'] = model_metrics_df['epoch'] * sample_size / batch_size

    model_metrics_df = model_metrics_df[model_metrics_df['changes_in_directions'].notna()]
    ax.plot(model_metrics_df['iterations'].values, model_metrics_df['changes_in_directions'].values, 
            label='changes in directions', c='r', linestyle='--')
    ax.set_yscale('log')
    ax.legend(prop={'size': 16}, loc='upper left')

def plot_loss_gradient_inner_product_to_norm_ratio(ax, model_metrics, epoch, sample_size, batch_size, regularization=None, *args, **kwargs):
    ax.clear()
    title = f'Global optimality condition 1 (Epoch = {epoch})'
    ax.set_xlabel(('S' if batch_size < sample_size else '') + 'GD steps')
    model_metrics_df = pandas.DataFrame(model_metrics)
    model_metrics_df['iterations'] = model_metrics_df['epoch'] * sample_size / batch_size

    #if 'Lambda' in model_metrics_df:
    #    df = model_metrics_df[model_metrics_df['Lambda'].notna()]
    #    plot_series_and_reference_on_ax(ax, df['iterations'].values, df['Lambda'].values, label='$\Lambda$', color='red', lower_lim=-1, log_scale=True)
        
    model_metrics_df = model_metrics_df[model_metrics_df['loss_gradient_inner_product_to_norm_ratio_avg'].notna()]
    plot_series_and_reference_on_ax(ax, model_metrics_df['iterations'].values, model_metrics_df['loss_gradient_inner_product_to_norm_ratio_avg'].values, 
                                    fill_between_y=model_metrics_df['loss_gradient_inner_product_to_norm_ratio_std'].values, 
                                    lower_bound=model_metrics_df['loss_gradient_inner_product_to_norm_ratio_min'].values, 
                                    upper_bound=model_metrics_df['loss_gradient_inner_product_to_norm_ratio_max'].values,
                                    label=f'$\langle-\\nabla L(f(X)), v_j \phi(w_j^\\top X)\\rangle \, / \, \|v_j\|\|w_j\|$', 
                                    lower_lim=-1, log_scale=True)
    
    if 'regularization' in model_metrics_df:
        df = model_metrics_df[model_metrics_df['regularization'].notna()]
        regularization = df['regularization'].values[-1]
        plot_series_and_reference_on_ax(ax, df['iterations'].values, df['regularization'].values, label=f'$\lambda$ = {regularization:.4f}', 
                                        color='black', log_scale=True)
    ax.legend(prop={'size': 16}, loc='upper right')
    ax.set_title(title)

def execute_experiment(train_data, test_data, model_class, seed:int, epochs:int, learning_rate:float, 
                       regularization:float=0, model=None, train_loss_class=torch.nn.BCEWithLogitsLoss, 
                       saving_epochs_interval=1, callbacks_epochs_interval=0, 
                       train_data_callbacks=None, test_data_callbacks=None, pruning_callback=None,
                       callbacks_epochs=[], callbacks=[], overwrite=False, **experiment):
    experiment.update(dict(seed=seed, epochs=epochs, learning_rate=learning_rate, regularization=regularization))
    device, generator = initialize(seed)
    if model is None:
        if not overwrite and experiment_exists(**experiment):
            experiment = {'epoch': 0, **load_experiment(**experiment), **experiment}
            model = model_class.load(**experiment).to(device)
            generator = set_random_states(**experiment)

        else: 
            model = model_class(**experiment).to(device)

    train_data.generator = test_data.generator = generator
    if regularization > 0:
        class RegularizedLoss(train_loss_class):
            def __call__(self, *args, **kwds):
                return super().__call__(*args, **kwds) + regularization * torch.sum(model.norms)
            
        train_loss_class = RegularizedLoss
    
    train_loss = train_loss_class()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    if 'train_loss' not in experiment:
        experiment.update(dict(epoch=0, train_time=[0], model_metrics=[]))
        experiment.update(dict(test_accuracy=[test(test_data, model, accuracy, device)]))
        experiment.update(dict(train_accuracy=[test(train_data, model, accuracy, device)]))
        experiment.update(dict(test_loss=[test(test_data, model, train_loss, device, callbacks=test_data_callbacks, **experiment)]))
        experiment.update(dict(train_loss=[test(train_data, model, train_loss, device, calculate_gradients=True, retain_graph=True, 
                                                callbacks=train_data_callbacks, **experiment)]))

    else:
        test(test_data, model, train_loss, device, callbacks=test_data_callbacks, **experiment)
        test(train_data, model, train_loss, device, calculate_gradients=True, retain_graph=True, callbacks=train_data_callbacks, **experiment)

    for callback in callbacks: callback(model=model, train_data=train_data, test_data=test_data, **experiment)
    for epoch in range(experiment['epoch'] + 1, epochs + 1):
        start_time = time.time()
        train(train_data, model, train_loss, optimizer, device)
        end_time = time.time()
        train_time = experiment['train_time'][-1] + end_time - start_time
        if pruning_callback:
            model_pruned = pruning_callback(model, train_data=train_data, test_data=test_data, **experiment)
            if model_pruned:
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        experiment['epoch'] = epoch#model.epoch = model.epoch + 1
        experiment['train_time'].append(train_time)
        experiment['test_accuracy'].append(test(test_data, model, accuracy, device))
        experiment['train_accuracy'].append(test(train_data, model, accuracy, device))
        experiment['test_loss'].append(test(test_data, model, train_loss, device, callbacks=test_data_callbacks, **experiment))
        experiment['train_loss'].append(test(train_data, model, train_loss, device, callbacks=train_data_callbacks, 
                                             calculate_gradients=True, retain_graph=True, **experiment))
        experiment['random_states'] = get_random_states()
        
        #if epoch % saving_epochs_interval == 0 or epoch == epochs:
        #    model.save()
        #    save_experiment(experiment)
        
        if (callbacks_epochs_interval and epoch % callbacks_epochs_interval == 0) or epoch in callbacks_epochs or epoch == epochs:
            #experiment['model_metrics'] += model.metrics
            for callback in callbacks: 
                callback(model=model, train_data=train_data, test_data=test_data, **experiment)

    return experiment, model, device, generator
