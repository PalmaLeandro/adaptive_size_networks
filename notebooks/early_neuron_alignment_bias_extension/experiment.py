import sys, os, time, torch

sys.path.append(os.path.abspath(os.path.join('..')))  # Allow repository modules to be imported

from utils.optimization import initialize, train, test, accuracy, get_random_states, set_random_states
from utils.persistance import file_name_from_parameters, experiment_exists, load_experiment, save_experiment

def execute_experiment(train_data, test_data, model_class, seed:int, epochs:int, learning_rate:float, convergence_epsilon=0., 
                       growing_epochs_interval=1, saving_epochs_interval=1, callbacks_epochs_interval=0, callbacks_epochs=[],
                       callbacks=[], override=False, experiment_name_parameters=None, **experiment):
    experiment.update(dict(seed=seed, epochs=epochs, learning_rate=learning_rate, convergence_epsilon=convergence_epsilon))
    experiment.update(dict(epoch=0, id=file_name_from_parameters(experiment_name_parameters, **experiment)))
    device, generator = initialize(seed)
    if not override and experiment_exists(**experiment, experiment_name_parameters=experiment_name_parameters):
        experiment = load_experiment(**experiment, experiment_name_parameters=experiment_name_parameters)
        model = model_class.load(**experiment).to(device)
        generator = set_random_states(**experiment)

    else: 
        model = model_class(**experiment).to(device)

    train_data.generator = test_data.generator = generator
    train_loss = torch.nn.CrossEntropyLoss() if model.classes > 2 else torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    if 'train_loss' not in experiment:
        experiment.update(dict(
            train_time=[0],
            test_accuracy=[test(test_data, model, accuracy, device)],
            train_accuracy=[test(train_data, model, accuracy, device)],
            test_loss=[test(test_data, model, train_loss, device)],
            train_loss=[test(train_data, model, train_loss, device, calculate_gradients=True, retain_graph=True)],
            architecture=model.architecture,
            model_metrics=[{**model_metrics, 'epoch': 0} for model_metrics in model.metrics]
        ))

    training_epochs = len(experiment['train_loss'])
    for callback in callbacks: 
        callback(model=model, train_data=train_data, test_data=test_data, **experiment)

    for epoch in range(training_epochs, epochs + 1):
        start_time = time.time()
        train(train_data, model, train_loss, optimizer, device)
        end_time = time.time()
        train_time = experiment['train_time'][-1] + end_time - start_time
        experiment['epoch'] = model.epoch = model.epoch + 1
        experiment['train_loss'].append(test(train_data, model, train_loss, device, calculate_gradients=True, retain_graph=True))
        experiment['model_metrics'] += [{**model_metrics, 'epoch': epoch} for model_metrics in model.metrics]
        experiment['train_time'].append(train_time)
        experiment['test_loss'].append(test(test_data, model, train_loss, device))
        experiment['train_accuracy'].append(test(train_data, model, accuracy, device))
        experiment['test_accuracy'].append(test(test_data, model, accuracy, device))
        experiment['random_states'] = get_random_states()
        if epoch % growing_epochs_interval == 0 or epoch == epochs:
            if abs(experiment['train_loss'][-1] - experiment['train_loss'][-2]) < convergence_epsilon:
                print(f'Convergence achieve according to convergence_epsilon = {convergence_epsilon}')
                architecture_change = model.prune() or model.grow()
                if architecture_change:
                    experiment['architecture'] = model.architecture
                    experiment['model_metrics'] += [{'architecture_change': architecture_change, 'epoch': epoch}]
                    optimizer.zero_grad()
                    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        
        if epoch % saving_epochs_interval == 0 or epoch == epochs:
            model.save()
            save_experiment(experiment, experiment_name_parameters)
        
        if (callbacks_epochs_interval and epoch % callbacks_epochs_interval == 0) or epoch in callbacks_epochs or epoch == epochs:
            for callback in callbacks: 
                callback(model=model, train_data=train_data, test_data=test_data, **experiment)
