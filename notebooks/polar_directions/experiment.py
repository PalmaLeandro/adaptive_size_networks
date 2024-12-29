import sys, os, time, numpy, pandas, torch

sys.path.append(os.path.abspath(os.path.join('..')))  # Allow repository modules to be imported

from utils.optimization import initialize, train, test, accuracy, get_random_states, set_random_states
from utils.persistance import experiment_exists, load_experiment, save_experiment
from utils.plots import plot_series_and_reference_on_ax

def model_summary(model, model_metrics, epoch, bias, *args, **kwargs):
    weights_products = model.weights_products.detach().cpu().numpy()
    inner_products = torch.tensor([- model.layers[0].weight.grad[neuron_index] @ model.layers[0].weight[neuron_index] 
                                   for neuron_index in range(model.architecture[0])]).abs().detach().cpu().numpy()
    inner_products_to_weights_products_ratio = inner_products / weights_products
    model_metrics.append({'epoch': epoch, 'inner_products_to_weights_products_ratio': inner_products_to_weights_products_ratio.tolist()})
    for layer_index, layer in enumerate(model.layers):
        layer_metrics = {'epoch': epoch, 'layer': layer_index, 'hidden_units': layer.weight.shape[0],
                         'neurons_weights_norm': layer.weight.norm(dim=1).detach().cpu().tolist()}
        if bias: layer_metrics.update({'biases': layer.bias.detach().cpu().tolist()})
        model_metrics.append(layer_metrics)
        
    model_metrics.append({'epoch': epoch, 'layer': len(model.layers), 'hidden_units': model.output_layer.weight.shape[1],
                          'neurons_weights_norm': model.output_layer.weight.abs().squeeze(dim=0).detach().cpu().tolist()})

def plot_neurons_inner_product_to_weights_products(ax, model_metrics, epoch, sample_size, batch_size, regularization=None, aggregate_neurons=True, *args, **kwargs):
    ax.clear()
    ax.set_title(f'Inner products to Weights products ratio (Epoch = {epoch})')
    ax.set_xlabel('iterations')
    model_metrics_df = pandas.DataFrame(model_metrics)
    model_metrics_df = model_metrics_df[model_metrics_df['inner_products_to_weights_products_ratio'].apply(lambda x: x is not numpy.nan)]

    model_metrics_df['iterations'] = model_metrics_df['epoch'] * sample_size / batch_size
    if aggregate_neurons:
        ratio_average = model_metrics_df.apply(lambda row: numpy.average(row['inner_products_to_weights_products_ratio']) / (regularization or 1.), axis='columns').to_numpy()
        ratio_stddev = model_metrics_df.apply(lambda row: numpy.std(row['inner_products_to_weights_products_ratio']) / (regularization or 1.), axis='columns').to_numpy()
        ratio_min = model_metrics_df.apply(lambda row: numpy.min(row['inner_products_to_weights_products_ratio']) / (regularization or 1.), axis='columns').to_numpy()
        ratio_max = model_metrics_df.apply(lambda row: numpy.max(row['inner_products_to_weights_products_ratio']) / (regularization or 1.), axis='columns').to_numpy()
        #y_min, y_max = min(*ratio_min.tolist(), regularization), max(*ratio_max.tolist(), regularization)
        plot_series_and_reference_on_ax(ax, model_metrics_df['iterations'].tolist(), ratio_average, 
                                        fill_between_y=ratio_stddev, lower_bound=ratio_min, upper_bound=ratio_max,
                                        label=('$\lambda \,$' if regularization else '') + f'$E |<∂l(f(X)), v_j \phi(w_j^t X)>| \, / \, |v_j|||w_j||$' + ('$\lambda \,$' if regularization else ''))
    else:
        for neuron_index in range(len(model_metrics_df['inner_products_to_weights_products_ratio'].tolist()[-1])):
            plot_series_and_reference_on_ax(ax, model_metrics_df['iterations'].tolist(), 
                                            model_metrics_df.apply(lambda row: row['inner_products_to_weights_products_ratio'][neuron_index] / (regularization or 1.), 
                                                                   axis='columns').to_numpy(), label=f'$E |<∂l(f(X)), v_{neuron_index} \phi(w_{neuron_index}^t X)>| \, / \, |v_{neuron_index}|||w_{neuron_index}||$' +
                                                                   ('$\lambda \,$' if regularization else ''))
        #y_min = min(*model_metrics_df.apply(lambda row: min(row['inner_products_to_weights_products_ratio']), axis='columns').tolist(), regularization)
        #y_max = max(*model_metrics_df.apply(lambda row: max(row['inner_products_to_weights_products_ratio']), axis='columns').tolist(), regularization)

    ax.legend()
    ax.set_yscale('log')
    #ax.set_ylim(y_min, y_max)

def execute_experiment(train_data, test_data, model_class, seed:int, epochs:int, learning_rate:float, 
                       regularization:float=0, train_loss_class=torch.nn.BCEWithLogitsLoss, 
                       saving_epochs_interval=1, callbacks_epochs_interval=0, 
                       callbacks_epochs=[], callbacks=[], overwrite=False, **experiment):
    experiment.update(dict(seed=seed, epochs=epochs, learning_rate=learning_rate, regularization=regularization))
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
                return super().__call__(*args, **kwds) + regularization * torch.sum(model.weights_products)
            
        train_loss_class = RegularizedBCEWithLogitsLoss
    
    train_loss = train_loss_class()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    if 'train_loss' not in experiment:
        experiment.update(dict(
            epoch=0,
            train_time=[0],
            test_accuracy=[test(test_data, model, accuracy, device)],
            train_accuracy=[test(train_data, model, accuracy, device)],
            test_loss=[test(test_data, model, train_loss, device)],
            train_loss=[test(train_data, model, train_loss, device, calculate_gradients=True, retain_graph=True)],
            architecture=model.architecture,
            model_metrics=model.metrics
        ))

    else:
        test(train_data, model, train_loss, device, calculate_gradients=True, retain_graph=True)

    for callback in callbacks: callback(model=model, train_data=train_data, test_data=test_data, **experiment)
    for epoch in range(experiment['epoch'] + 1, epochs + 1):
        start_time = time.time()
        train(train_data, model, train_loss, optimizer, device)
        end_time = time.time()
        train_time = experiment['train_time'][-1] + end_time - start_time
        experiment['epoch'] = model.epoch = model.epoch + 1
        experiment['train_time'].append(train_time)
        experiment['test_accuracy'].append(test(test_data, model, accuracy, device))
        experiment['train_accuracy'].append(test(train_data, model, accuracy, device))
        experiment['test_loss'].append(test(test_data, model, train_loss, device))
        experiment['train_loss'].append(test(train_data, model, train_loss, device, calculate_gradients=True, retain_graph=True))
        experiment['random_states'] = get_random_states()
        
        if epoch % saving_epochs_interval == 0 or epoch == epochs:
            model.save()
            save_experiment(experiment)
        
        if (callbacks_epochs_interval and epoch % callbacks_epochs_interval == 0) or epoch in callbacks_epochs or epoch == epochs:
            experiment['model_metrics'] += model.metrics
            for callback in callbacks: 
                callback(model=model, train_data=train_data, test_data=test_data, **experiment)

    return experiment, model, device, generator