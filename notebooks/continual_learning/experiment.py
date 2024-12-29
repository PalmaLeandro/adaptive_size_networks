import sys, os, time, torch

sys.path.append(os.path.abspath(os.path.join('..')))  # Allow repository modules to be imported

from utils.optimization import initialize, train, test, accuracy, get_random_states, set_random_states
from utils.persistance import experiment_exists, load_experiment, save_experiment
from utils.plots import plot_series_and_reference_on_ax

def extract_samples(train_data, filter_classes=[], *args, **kwargs):
    inputs = []; labels = []
    for batch_inputs, batch_labels in train_data:
        if filter_classes:
            filtered_indices = torch.isin(torch.argmax(batch_labels, dim=1), torch.tensor(filter_classes)).nonzero().flatten()
            batch_inputs, batch_labels = batch_inputs[filtered_indices], batch_labels[filtered_indices]
            batch_labels = batch_labels[:, :max(filter_classes) + 1]
            
        inputs.append(batch_inputs); labels.append(batch_labels)
    
    return torch.concatenate(inputs), torch.concatenate(labels)

def execute_experiment(train_data, test_data, model_class, seed:int, epochs:int, learning_rate:float, 
                       new_classes_per_iteration, classes, train_loss_class=torch.nn.BCEWithLogitsLoss, 
                       saving_epochs_interval=1, callbacks_epochs_interval=0, callbacks_epochs=[], 
                       callbacks=[], overwrite=False, prune=True, **experiment):
    experiment.update(dict(seed=seed, epochs=epochs, learning_rate=learning_rate, 
                           new_classes_per_iteration=new_classes_per_iteration, classes=classes))
    device, generator = initialize(seed)
    if not overwrite and experiment_exists(**experiment):
        experiment = {'epoch': 0, **load_experiment(**experiment), **experiment}
        model = model_class.load(**experiment).to(device)
        generator = set_random_states(**experiment)

    else: 
        model = model_class(**experiment).to(device)

    train_data.generator = test_data.generator = generator
    train_loss = train_loss_class()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    if 'train_loss' not in experiment:
        experiment.update(dict(epoch=0, train_time=[0], model_metrics=[]))
        experiment.update(dict(test_accuracy=[test(test_data, model, accuracy, device)]))
        experiment.update(dict(train_accuracy=[test(train_data, model, accuracy, device)]))
        experiment.update(dict(test_loss=[test(test_data, model, train_loss, device, **experiment)]))
        experiment.update(dict(train_loss=[test(train_data, model, train_loss, device, calculate_gradients=True, retain_graph=True, **experiment)]))

    else:
        test(train_data, model, train_loss, device, calculate_gradients=True, retain_graph=True, **experiment)

    for callback in callbacks: callback(model=model, train_data=train_data, test_data=test_data, **experiment)
    classes_labels = list(range(classes))
    for learning_iteration in range((classes // new_classes_per_iteration) + 1):
        classes_to_learn = classes_labels[learning_iteration * new_classes_per_iteration:min((learning_iteration + 1) * new_classes_per_iteration, classes)]
        print(f'Learning iteration {learning_iteration}. Classes to learn: {classes_to_learn}')
        if not classes_to_learn: break
        for epoch in range(max(1, learning_iteration * epochs), epochs * (learning_iteration + 1) + 1):
            start_time = time.time()
            train(train_data, model, train_loss, optimizer, device, filter_classes=classes_to_learn)
            end_time = time.time()
            train_time = experiment['train_time'][-1] + end_time - start_time
            experiment['epoch'] = model.epoch = model.epoch + 1
            experiment['learning_iteration'] = learning_iteration
            experiment['train_time'].append(train_time)
            experiment['test_accuracy'].append(test(test_data, model, accuracy, device, filter_classes=classes_to_learn))
            experiment['train_accuracy'].append(test(train_data, model, accuracy, device, filter_classes=classes_to_learn))
            experiment['test_loss'].append(test(test_data, model, train_loss, device, **experiment, filter_classes=classes_to_learn))
            experiment['train_loss'].append(test(train_data, model, train_loss, device, calculate_gradients=True, retain_graph=True, **experiment, filter_classes=classes_to_learn))
            experiment['random_states'] = get_random_states()
            
            #if epoch % saving_epochs_interval == 0 or epoch == epochs:
            #    model.save()
            #    save_experiment(experiment)
            
            if (callbacks_epochs_interval and epoch % callbacks_epochs_interval == 0) or epoch in callbacks_epochs or epoch % epochs == 0:
                for callback in callbacks: 
                    callback(model=model, train_data=train_data, test_data=test_data, **experiment, filter_classes=classes_to_learn)

        if prune:
            model.prune(*extract_samples(train_data, filter_classes=classes_to_learn))
            
        for callback in callbacks: 
            callback(model=model, train_data=train_data, test_data=test_data, **experiment)

        if model.n_observed_classes < classes:
            model.increase_width()
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    return experiment, model, device, generator
