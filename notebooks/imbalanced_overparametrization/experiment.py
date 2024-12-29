import os, sys, time, torch

sys.path.append(os.path.abspath(os.path.join('..')))  # Allow repository modules to be imported

from utils.optimization import initialize, train, test
from utils.persistance import save_experiment, PersistableModel
from settings.noisy_sigmoid_teacher_network import get_dataloader

EXPERIMENT_PARAMETERS = ['seed', 'input_dimension', 'sample_size', 'batch_size', 'hidden_units',
                         'epochs', 'learning_rate', 'noise_scale', 'initialization_variance']

def execute_experiment(seed, input_dimension, sample_size, batch_size, hidden_units, epochs, learning_rate, 
                       noise_scale, initialization_variance, runs_per_model, save_models_path=None, save_experiments_path=None, 
                       plot_results=False, plot_results_on_canvas=None, saving_epochs_interval=1, verbose=False, second_axis=None):
    device = initialize(seed)
    train_loader, output_weights = get_dataloader(input_dimension, sample_size, batch_size, noise_scale)
    test_loader = get_dataloader(input_dimension, sample_size, batch_size, noise_scale, output_weights)
    experiment = {
        'seed': seed,
        'input_dimension': input_dimension,
        'sample_size': sample_size,
        'batch_size': batch_size,
        'hidden_units': hidden_units,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'distinction': 'run',
        'noise_scale': noise_scale,
        'initialization_variance': initialization_variance,
        'train': 'MSE',
        'test': 'MSE',
        'train_time': 'seconds',
        'models_runs': []
    }

    if plot_results or plot_results_on_canvas: from utils.plotting import plot_experiment
    for run_number in range(runs_per_model):
        if verbose: print(f'Run {run_number}')

        model = ImbalancedOverparametrizationNeuralNet(**experiment, run=run_number).to(device)
        loss = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        run = {
            'distinction': run_number,
            'train': [test(train_loader, model, loss, device)],
            'train_time': [0],
            'test': [test(test_loader, model, loss, device, verbose=verbose)],
            'imbalance': [model.imbalance],
            'input_layer_weights_norm': [model.input_layer_weights_norm],
            'output_layer_weights_norm': [model.output_layer_weights_norm]
        }
        experiment['models_runs'].append(run)
        for epoch in range(1, epochs + 1):
            if verbose: print(f'Epoch {epoch}')
                
            start_time = time.time()
            train_loss = train(train_loader, model, loss, optimizer, device, verbose=verbose)
            end_time = time.time()
            train_time = run['train_time'][-1] + end_time - start_time
            test_loss = test(test_loader, model, loss, device, verbose=verbose)
            run['train'].append(train_loss); run['train_time'].append(train_time); run['test'].append(test_loss)
            run['imbalance'].append(model.imbalance)
            run['input_layer_weights_norm'].append(model.input_layer_weights_norm)
            run['output_layer_weights_norm'].append(model.output_layer_weights_norm)

            if epoch % saving_epochs_interval == 0 or epoch == epochs:
                if save_models_path: model.save(save_models_path)
                if save_experiments_path: save_experiment(experiment, save_experiments_path, EXPERIMENT_PARAMETERS)

            if plot_results_on_canvas: plot_experiment(experiment, plot_results_on_canvas, second_axis=second_axis)
            
    if plot_results: plot_experiment(experiment, second_axis=second_axis)
    return experiment

    
class ImbalancedOverparametrizationNeuralNet(torch.nn.Module, PersistableModel):
    
    MODEL_NAME_PARAMETERS = EXPERIMENT_PARAMETERS + ['hidden_units', 'initialization_variance', 'run']

    def __init__(self, input_dimension:int, hidden_units:int, initialization_variance:float, *args, **kwargs):
        super(ImbalancedOverparametrizationNeuralNet, self).__init__()

        self.input_layer = torch.nn.Linear(input_dimension, hidden_units, bias=False)
        self.output_layer = torch.nn.Linear(hidden_units, input_dimension, bias=False)
        
        torch.nn.init.normal_(self.input_layer.weight, std=initialization_variance ** 0.5)
        torch.nn.init.normal_(self.output_layer.weight, std=initialization_variance ** 0.5)

        self.store_parameters(input_dimension=input_dimension, 
                              hidden_units=hidden_units, 
                              initialization_variance=initialization_variance,
                              **kwargs)
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.output_layer(x)
        return x
    
    @property
    def imbalance(self):
        return torch.norm(torch.matmul(self.input_layer.weight.t(), self.input_layer.weight) - 
                          torch.matmul(self.output_layer.weight, self.output_layer.weight.t())).detach().cpu().item()
    
    @property
    def input_layer_weights_norm(self):
        return torch.norm(self.input_layer.weight).detach().cpu().item()
    
    @property
    def output_layer_weights_norm(self):
        return torch.norm(self.output_layer.weight).detach().cpu().item()