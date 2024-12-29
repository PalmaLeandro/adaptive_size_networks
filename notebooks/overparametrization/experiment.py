import os, sys, time, torch

sys.path.append(os.path.abspath(os.path.join('..')))  # Allow repository modules to be imported

from utils.optimization import initialize, train, test
from utils.persistance import save_experiment, PersistableModel
from settings.two_layer_teacher_network import get_dataloader

EXPERIMENT_PARAMETERS = ['seed', 'input_dimension', 'sample_size', 'batch_size', 'epochs', 'learning_rate']

def execute_experiment(seed, input_dimension, sample_size, batch_size, epochs, learning_rate, 
                       models_overparametrizations, runs_per_model, save_models_path=None, save_experiments_path=None, 
                       plot_results=False, plot_results_on_canvas=None, saving_epochs_interval=1, verbose=False):
    device = initialize(seed)
    train_loader, true_output_weights = get_dataloader(input_dimension, sample_size, batch_size)
    test_loader = get_dataloader(input_dimension, sample_size, batch_size, true_output_weights)
    experiment = {
        'seed': seed,
        'input_dimension': input_dimension,
        'sample_size': sample_size,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'distinction': 'overparametrization',
        'true_output_weights': true_output_weights.tolist(),
        'models_overparametrizations': models_overparametrizations,
        'train': 'Mean Squared Error',
        'test': 'Mean Squared Error',
        'train_time': 'seconds',
        'models_runs': []
    }

    if plot_results or plot_results_on_canvas: from utils.plotting import plot_experiment
    
    for overparametrization in models_overparametrizations:
        if verbose: print(f'x{overparametrization} Oveparametrization')
    
        for run_number in range(runs_per_model):
            if verbose: print(f'Run {run_number}')

            model_parameters = {**experiment, 'overparametrization':overparametrization, 'run': run_number}
            model = OverparametrizationTwoLayerNeuralNet(**model_parameters).to(device)
            loss = torch.nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            run = {
                'distinction': overparametrization,
                'train': [test(train_loader, model, loss, device)],
                'train_time': [0],
                'test': [test(test_loader, model, loss, device, verbose=verbose)]
            }
            experiment['models_runs'].append(run)
            
            for epoch in range(1, epochs + 1):
                if verbose: print(f'Epoch {epoch}')
                    
                start_time = time.time()
                train_loss = train(train_loader, model, loss, optimizer, device, verbose=verbose)
                end_time = time.time()
                train_time = run['train_time'][-1] + end_time - start_time
                test_loss = test(test_loader, model, loss, device, verbose=verbose)
                run['train'].append(train_loss), run['train_time'].append(train_time), run['test'].append(test_loss)
            
                if epoch % saving_epochs_interval == 0 or epoch == epochs:
                    if save_models_path: model.save(save_models_path)
                    if save_experiments_path: save_experiment(experiment, save_experiments_path, EXPERIMENT_PARAMETERS)

                if plot_results_on_canvas: plot_experiment(experiment, plot_results_on_canvas)
                
    if plot_results: plot_experiment(experiment)
    return experiment

    
class OverparametrizationTwoLayerNeuralNet(torch.nn.Module, PersistableModel):
    
    MODEL_NAME_PARAMETERS = EXPERIMENT_PARAMETERS + ['overparametrization', 'run']

    def __init__(self, input_dimension:int, output_dimension:int, overparametrization:int, *args, **kwargs):
        super(OverparametrizationTwoLayerNeuralNet, self).__init__()
        hidden_units = int(input_dimension * overparametrization)

        self.linear1 = torch.nn.Linear(input_dimension, hidden_units, bias=False)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_units, 1, bias=False)

        self.store_parameters(input_dimension=input_dimension, 
                              overparametrization=overparametrization,
                              **kwargs)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x
