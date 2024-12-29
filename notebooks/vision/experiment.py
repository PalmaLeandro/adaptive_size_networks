import os, sys, time, torch

sys.path.append(os.path.abspath(os.path.join('..')))  # Allow repository modules to be imported

from utils.optimization import initialize, train, test, accuracy
from utils.persistance import save_experiment, PersistableModel
from settings.cifar10 import train_dataloader, test_dataloader, WIDTH, HEIGHT, CHANNELS, CLASSES, SAMPLE_SIZE

EXPERIMENT_PARAMETERS = ['dataset', 'seed', 'sample_size', 'batch_size', 'epochs', 'learning_rate']

def execute_experiment(seed, batch_size, epochs, learning_rate, channels_settings, runs_per_model, 
                       data_path=None, save_models_path=None, save_experiments_path=None, 
                       plot_results=False, plot_results_on_canvas=None, saving_epochs_interval=1, verbose=False):
    device = initialize(seed)
    train_loader, test_loader = train_dataloader(batch_size, data_path), test_dataloader(batch_size, data_path)
    experiment = {
        'dataset': 'CIFAR10',
        'seed': seed,
        'sample_size': SAMPLE_SIZE,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'distinction': 'channels',
        'channels_settings': channels_settings,
        'train': 'Cross Entropy',
        'test': 'Accuracy',
        'train_time': 'seconds',
        'models_runs': []
    }
    if plot_results or plot_results_on_canvas: from utils.plotting import plot_experiment
    
    for channels in channels_settings:
        if verbose: print(f'{channels} channels')
    
        for run_number in range(runs_per_model):
            if verbose: print(f'Run {run_number}')

            model = VisionOverparametrizationTwoLayerNeuralNet(channels, **experiment, run=run_number).to(device)
            train_loss_fn, test_loss_fn = torch.nn.CrossEntropyLoss(), accuracy
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            run = {
                'distinction': channels,
                'train': [test(train_loader, model, train_loss_fn, device)],
                'train_time': [0],
                'test': [test(test_loader, model, test_loss_fn, device, verbose=verbose)]
            }
            experiment['models_runs'].append(run)
            for epoch in range(1, epochs + 1):
                if verbose: print(f'Epoch {epoch}')
                    
                start_time = time.time()
                train_loss = train(train_loader, model, train_loss_fn, optimizer, device, verbose=verbose)
                end_time = time.time()
                train_time = run['train_time'][-1] + end_time - start_time
                test_loss = test(test_loader, model, test_loss_fn, device, verbose=verbose)
                run['train'].append(train_loss), run['train_time'].append(train_time), run['test'].append(test_loss)
            
                if (epoch % saving_epochs_interval == 0 or epoch == epochs):
                    if save_models_path: model.save(save_models_path)
                    if save_experiments_path: save_experiment(experiment, save_experiments_path, EXPERIMENT_PARAMETERS)

                if plot_results_on_canvas: plot_experiment(experiment, plot_results_on_canvas)
    
    if plot_results: plot_experiment(experiment)
    return experiment

    
class VisionOverparametrizationTwoLayerNeuralNet(torch.nn.Module, PersistableModel):

    MODEL_NAME_PARAMETERS = EXPERIMENT_PARAMETERS + ['channels', 'run']

    def __init__(self, channels:int, *args, **kwargs):
        super(VisionOverparametrizationTwoLayerNeuralNet, self).__init__()

        self.conv = torch.nn.Conv2d(CHANNELS, channels, kernel_size=3, padding=1)
        self.activation = torch.nn.MaxPool2d(kernel_size=(WIDTH, HEIGHT), stride=1)
        self.fully_conected = torch.nn.Conv2d(channels, CLASSES, kernel_size=1)
        
        self.store_parameters(channels=channels, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.fully_conected(x)
        return x.reshape(-1, CLASSES)
