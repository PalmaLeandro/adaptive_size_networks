import os, sys, time, torch

sys.path.append(os.path.abspath(os.path.join('..')))  # Allow repository modules to be imported

from utils.optimization import initialize, train, test, accuracy
from utils.persistance import save_experiment, PersistableModel
from settings.noisy_xor import get_dataloader

EXPERIMENT_PARAMETERS = ['seed', 'input_dimension', 'sample_size', 'batch_size', 'epochs', 'learning_rate', 
                         'noise_rate', 'within_cluster_variance', 'hidden_units', 'initialization_variance']

def execute_experiment(seed, noise_rate, within_cluster_variance, input_dimension, sample_size, batch_size, epochs, 
                       learning_rate, hidden_units, initialization_variance, runs_per_model, 
                       save_models_path=None, save_experiments_path=None, plot_results=False, 
                       plot_results_on_canvas=None, saving_epochs_interval=1, verbose=False, callback=None, **kwargs):
    device = initialize(seed)
    train_data, rotation_matrix = get_dataloader(noise_rate, within_cluster_variance, input_dimension, sample_size, batch_size)
    test_data = get_dataloader(noise_rate, within_cluster_variance, input_dimension, sample_size, batch_size, rotation_matrix)
    experiment = {
        'seed': seed,
        'input_dimension': input_dimension,
        'sample_size': sample_size,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'noise_rate': noise_rate,
        'within_cluster_variance': within_cluster_variance,
        'hidden_units': hidden_units,
        'initialization_variance': initialization_variance,
        'distinction': 'run',
        'train': 'Accuracy',
        'test': 'Accuracy',
        'train_time': 'seconds',
        'models_runs': []
    }
    if plot_results or plot_results_on_canvas: from utils.plotting import plot_experiment

    for run_number in range(runs_per_model):
        if verbose: print(f'Run {run_number}')

        model = PrunableTwoLayerNeuralNet(**experiment, run=run_number).to(device)
        train_loss, test_loss = torch.nn.BCEWithLogitsLoss(), accuracy
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        run = {
            'distinction': run_number,
            'train': [test(train_data, model, test_loss, device)],
            'train_time': [0],
            'test': [test(test_data, model, test_loss, device, verbose=verbose)]
        }
        experiment['models_runs'].append(run)
        if callback: callback(model=model, epoch=0, rotation_matrix=rotation_matrix, **experiment, **kwargs)
        for epoch in range(1, epochs + 1):
            if verbose: print(f'Epoch {epoch}')
                
            start_time = time.time()
            train(train_data, model, train_loss, optimizer, device, verbose=verbose)
            end_time = time.time()
            train_time = run['train_time'][-1] + end_time - start_time

            train_loss_value = test(train_data, model, test_loss, device, verbose=False)
            test_loss_value = test(test_data, model, test_loss, device, verbose=verbose)
            
            run['train'].append(train_loss_value)
            run['train_time'].append(train_time)
            run['test'].append(test_loss_value)
        
            if epoch % saving_epochs_interval == 0 or epoch == epochs:
                if save_models_path: model.save(save_models_path)
                if save_experiments_path: save_experiment(experiment, save_experiments_path, EXPERIMENT_PARAMETERS)
                if callback: callback(model=model, epoch=epoch, rotation_matrix=rotation_matrix, **experiment, **kwargs)

            if plot_results_on_canvas: plot_experiment(experiment, plot_results_on_canvas)
                
    if plot_results: plot_experiment(experiment)
    return experiment

    
class PrunableTwoLayerNeuralNet(torch.nn.Module, PersistableModel):

    MODEL_NAME_PARAMETERS = EXPERIMENT_PARAMETERS + ['run']

    def __init__(self, input_dimension:int, hidden_units:int, initialization_variance:float, *args, **kwargs):
        super(PrunableTwoLayerNeuralNet, self).__init__()
        if hidden_units % 2 == 1:
            hidden_units -= 1
            print(f'Only even number of hidden units allowed. Switching to hidden units = {hidden_units}') 

        self._input_layer_weights = torch.nn.Parameter(torch.ones(input_dimension, hidden_units))
        self.pruning_mask = torch.ones(input_dimension, hidden_units)
        self.activation = torch.nn.ReLU()
        self.output_layer_weights = torch.Tensor([1] * (hidden_units // 2) + [-1] * (hidden_units // 2))
        self.output_layer_weights /= hidden_units ** 0.5 # scaling of the output
        
        torch.nn.init.normal_(self._input_layer_weights, std=initialization_variance ** 0.5)

        self.store_parameters(input_dimension=input_dimension, 
                              hidden_units=hidden_units, 
                              initialization_variance=initialization_variance,
                              **kwargs)
        
    def forward(self, x):
        x = torch.matmul(x, self.input_layer_weights)
        x = self.activation(x)
        x = torch.matmul(x, self.output_layer_weights)
        return x.unsqueeze(1)
    
    @property
    def input_layer_weights(self):
        return self.pruning_mask * self._input_layer_weights

    def to(self, device):
        super().to(device)
        self._input_layer_weights = self._input_layer_weights.to(device)
        self.pruning_mask = self.pruning_mask.to(device)
        self.output_layer_weights = self.output_layer_weights.to(device)
        return self
    
    def prune_weights_by_magnitude(self, pruning_raio):
        input_layer_weights = self.input_layer_weights.detach().cpu().numpy()
        weights_indices = [
            (neuron_index, weight_index, weight) 
            for neuron_index, neuron_weights in enumerate(input_layer_weights) 
            for weight_index, weight in enumerate(neuron_weights)
        ]
        weights_to_prune = sorted(weights_indices, key=lambda x: abs(x[2]))[:int(len(weights_indices) * pruning_raio)]
        weights_indices_to_prune = list(map(lambda x: (x[0], x[1]), weights_to_prune))
        for weight_indices_to_prune in weights_indices_to_prune: 
            self.pruning_mask[weight_indices_to_prune] = 0.
