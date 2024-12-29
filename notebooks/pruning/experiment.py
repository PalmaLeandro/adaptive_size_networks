import sys, os, time, numpy, pandas, torch

sys.path.append(os.path.abspath(os.path.join('..')))  # Allow repository modules to be imported

from utils.optimization import initialize, train, test, accuracy, get_random_states, set_random_states, extract_samples
from utils.persistance import experiment_exists, load_experiment, save_experiment
from utils.plots import plot_series_and_reference_on_ax

def model_summary(model, epoch, model_metrics=None, *args, **kwargs):
    weights_products = model.weights_products.detach().cpu().numpy()
    inner_products = torch.tensor([- model.layers[0].weight.grad[neuron_index] @ model.layers[0].weight[neuron_index] 
                                   for neuron_index in range(model.architecture[0])]).abs().detach().cpu().numpy()
    loss_gradient_inner_product_to_norm_ratio = (inner_products / weights_products).detach().cpu().tolist()
    model_metrics.append({'epoch': epoch, 'loss_gradient_inner_product_to_norm_ratio': loss_gradient_inner_product_to_norm_ratio,
                          'weights_products': weights_products.tolist()})

def save_margins(model, predictions, labels, model_metrics, epoch, *args, **kwargs):
    margins = predictions * ((labels * 2.) - 1.)
    min_margin = margins.min().detach().cpu()
    exp_margins_sum = (- predictions * labels).exp().sum().detach().cpu().item()
    norm = model.norm.detach().cpu().item()
    Lambda = ((min_margin ** (1. - (1. / 2.))) * norm / exp_margins_sum).detach().cpu().item()
    model_metrics.append({'epoch': epoch, 'min_margin': min_margin, 'exp_margins_sum': exp_margins_sum, 'norm': norm, 'Lambda': Lambda})

def plot_loss_gradient_inner_product_to_norm_ratio(ax, model_metrics, epoch, sample_size, batch_size, regularization=None, aggregate_neurons=True, *args, **kwargs):
    ax.clear()
    ax.set_title(f'Loss gradient inner product with neurons predictions (Epoch = {epoch})')
    ax.set_xlabel(('S' if batch_size < sample_size else '') + 'GD steps')
    model_metrics_df = pandas.DataFrame(model_metrics)
    model_metrics_df['iterations'] = model_metrics_df['epoch'] * sample_size / batch_size
    ax.plot(model_metrics_df['iterations'].tolist(), model_metrics_df['Lambda'].tolist(), c='r', linestyle='--', label='$\Lambda$')
    if regularization:
        ax.hlines(regularization,0,  epoch * sample_size / batch_size, color='k', label='$\hat{\Lambda}$')

    model_metrics_df = model_metrics_df[model_metrics_df['loss_gradient_inner_product_to_norm_ratio'].apply(lambda x: x is not numpy.nan)]
    if aggregate_neurons:
        ratio_average = model_metrics_df.apply(lambda row: numpy.average(row['loss_gradient_inner_product_to_norm_ratio']), axis='columns').to_numpy()
        ratio_stddev = model_metrics_df.apply(lambda row: numpy.std(row['loss_gradient_inner_product_to_norm_ratio']), axis='columns').to_numpy()
        ratio_min = model_metrics_df.apply(lambda row: numpy.min(row['loss_gradient_inner_product_to_norm_ratio']), axis='columns').to_numpy()
        ratio_max = model_metrics_df.apply(lambda row: numpy.max(row['loss_gradient_inner_product_to_norm_ratio']), axis='columns').to_numpy()
        plot_series_and_reference_on_ax(ax, model_metrics_df['iterations'].tolist(), ratio_average, 
                                        fill_between_y=ratio_stddev, lower_bound=ratio_min, upper_bound=ratio_max,
                                        label=('$\lambda \,$' if regularization else '') + f'$E |<∂l(f(X)), v_j \phi(w_j^t X)>| \, / \, |v_j|||w_j||$' + ('$\lambda \,$' if regularization else ''))
    else:
        for neuron_index in range(len(model_metrics_df['loss_gradient_inner_product_to_norm_ratio'].tolist()[-1])):
            plot_series_and_reference_on_ax(ax, model_metrics_df['iterations'].tolist(), 
                                            model_metrics_df.apply(lambda row: row['loss_gradient_inner_product_to_norm_ratio'][neuron_index], 
                                                                   axis='columns').to_numpy(), label=f'$E |<∂l(f(X)), v_{neuron_index} \phi(w_{neuron_index}^t X)>| \, / \, |v_{neuron_index}|||w_{neuron_index}||$')

    ax.legend()

def execute_experiment(train_data, test_data, model_class, seed:int, epochs:int, learning_rate:float, 
                       new_classes_per_iteration, classes, pruning_periodicity,
                       regularization:float=0, train_loss_class=torch.nn.BCEWithLogitsLoss, 
                       saving_epochs_interval=1, callbacks_epochs_interval=0, 
                       train_loss_callback=None, test_loss_callback=None,
                       callbacks_epochs=[], callbacks=[], overwrite=False, 
                       **experiment):
    experiment.update(dict(seed=seed, epochs=epochs, learning_rate=learning_rate, regularization=regularization,
                           new_classes_per_iteration=new_classes_per_iteration, classes=classes, pruning_periodicity=pruning_periodicity))
    device, generator = initialize(seed)
    if not overwrite and experiment_exists(**experiment):
        experiment = {'epoch': 0, **load_experiment(**experiment), **experiment}
        model = model_class.load(**experiment).to(device)
        generator = set_random_states(**experiment)

    else: 
        model = model_class(**experiment).to(device)

    train_data.generator = test_data.generator = generator
    if regularization > 0:
        class RegularizedBCEWithLogitsLoss(train_loss_class):
            def __call__(self, *args, **kwds):
                return super().__call__(*args, **kwds) + regularization * model.norm
            
        train_loss_class = RegularizedBCEWithLogitsLoss
    
    train_loss = train_loss_class()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    if 'train_loss' not in experiment:
        experiment.update(dict(epoch=0, train_time=[0], model_metrics=[]))
        experiment.update(dict(test_accuracy=[test(test_data, model, accuracy, device)]))
        experiment.update(dict(train_accuracy=[test(train_data, model, accuracy, device)]))
        experiment.update(dict(test_loss=[test(test_data, model, train_loss, device, callback=test_loss_callback, **experiment)]))
        experiment.update(dict(train_loss=[test(train_data, model, train_loss, device, calculate_gradients=True, retain_graph=True, 
                                                callback=train_loss_callback, **experiment)]))

    else:
        test(train_data, model, train_loss, device, calculate_gradients=True, retain_graph=True, callback=train_loss_callback, **experiment)

    for callback in callbacks: callback(model=model, train_data=train_data, test_data=test_data, **experiment)
    for pruning_iteration in range((epochs // pruning_periodicity) + 1):
        for epoch in range(max(1, pruning_iteration * pruning_periodicity), min((pruning_iteration + 1) * pruning_periodicity, epochs + 1)):
            start_time = time.time()
            train(train_data, model, train_loss, optimizer, device)
            end_time = time.time()
            train_time = experiment['train_time'][-1] + end_time - start_time
            experiment['epoch'] = model.epoch = model.epoch + 1
            experiment['pruning_iteration'] = pruning_iteration
            experiment['train_time'].append(train_time)
            experiment['test_accuracy'].append(test(test_data, model, accuracy, device))
            experiment['train_accuracy'].append(test(train_data, model, accuracy, device))
            experiment['test_loss'].append(test(test_data, model, train_loss, device, callback=test_loss_callback, **experiment))
            experiment['train_loss'].append(test(train_data, model, train_loss, device, calculate_gradients=True, retain_graph=True, 
                                                 callback=train_loss_callback, **experiment))
            experiment['random_states'] = get_random_states()
            
            #if epoch % saving_epochs_interval == 0 or epoch == epochs:
            #    model.save()
            #    save_experiment(experiment)
            
            if (callbacks_epochs_interval and epoch % callbacks_epochs_interval == 0) or epoch in callbacks_epochs or epoch == epochs:
                for callback in callbacks: 
                    callback(model=model, train_data=train_data, test_data=test_data, **experiment)

        model.prune(*extract_samples(train_data))
        test(train_data, model, train_loss, device, calculate_gradients=True, retain_graph=True)
        #model.increase_width()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        for callback in callbacks: 
            callback(model=model, train_data=train_data, test_data=test_data, **experiment)

    return experiment, model, device, generator