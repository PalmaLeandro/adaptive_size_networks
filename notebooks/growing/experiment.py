import sys, os, time, numpy, pandas, torch

sys.path.append(os.path.abspath(os.path.join('..')))  # Allow repository modules to be imported

from utils.optimization import initialize, train, test, accuracy, get_random_states, set_random_states
from utils.persistance import experiment_exists, load_experiment, save_experiment

def plot_samples_train_loss_value(ax, train_data, model, train_loss_class, train_losses_values, *args, **kwargs):
    ax.clear()
    ax.set_title('Loss per sample')
    ax.set_xlabel('iteration')
    ax.set_ylabel('sample')
    inputs = []; labels = []
    for batch_inputs, batch_labels in train_data: inputs.append(batch_inputs); labels.append(batch_labels)
    inputs, labels = torch.concatenate(inputs), torch.concatenate(labels)
    train_losses = train_loss_class(reduction='none')(model(inputs), labels)
    train_losses_values.append(train_losses.detach().cpu().numpy())
    ax.imshow(numpy.stack(train_losses_values).T, cmap='viridis', interpolation='none')



def execute_experiment(train_data, test_data, model_class, seed:int, epochs:int, learning_rate:float, classes,
                       regularization:float=0, train_loss_class=torch.nn.BCEWithLogitsLoss, 
                       saving_epochs_interval=1, callbacks_epochs_interval=0, 
                       callbacks_epochs=[], callbacks=[], overwrite=False, 
                       **experiment):
    experiment.update(dict(seed=seed, epochs=epochs, learning_rate=learning_rate, regularization=regularization, classes=classes))
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
            train_loss=[test(train_data, model, train_loss, device, calculate_gradients=True, retain_graph=True)]
        ))

    else:
        test(train_data, model, train_loss, device, calculate_gradients=True, retain_graph=True)

    for callback in callbacks: callback(model=model, train_data=train_data, test_data=test_data, train_loss_class=train_loss_class, **experiment)
    while experiment['test_accuracy'][-1] < 1.:
        for epoch in range(1, epochs + 1):
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
            
            #if epoch % saving_epochs_interval == 0 or epoch == epochs:
            #    model.save()
            #    save_experiment(experiment)
            
            if (callbacks_epochs_interval and epoch % callbacks_epochs_interval == 0) or epoch in callbacks_epochs or epoch == epochs:
                for callback in callbacks: 
                    callback(model=model, train_data=train_data, test_data=test_data, train_loss_class=train_loss_class, **experiment)
            
        if experiment['test_accuracy'][-1] < 1.:
            model.increase_width()
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    return experiment, model, device, generator