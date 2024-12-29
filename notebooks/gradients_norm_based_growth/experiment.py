import os, sys, time, functools, numpy, torch

sys.path.append(os.path.abspath(os.path.join('..')))  # Allow repository modules to be imported

from settings import concentric_spheres
from utils.optimization import initialize, train, test, accuracy
from utils.persistance import save_experiment, PersistableModel

EXPERIMENT_PARAMETERS = ['seed', 'input_dimension', 'spheres_dimension', 'sample_size', 'batch_size', 'epochs', 'learning_rate', 
                         'hidden_units', 'bias', 'margin']

def execute_experiment(seed, input_dimension, spheres_dimension, sample_size, batch_size, epochs, learning_rate, hidden_units, bias, margin,
                       save_models_path=None, save_experiments_path=None, plot_results=False, plot_results_on_canvas=None, 
                       saving_epochs_interval=1, verbose=False, callbacks=[], **kwargs):
    if plot_results or plot_results_on_canvas: from utils.plotting import plot_experiment
    device = initialize(seed)
    experiment = {
        'seed': seed,
        'input_dimension': input_dimension,
        'spheres_dimension': spheres_dimension,
        'sample_size': sample_size,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'hidden_units': hidden_units,
        'bias': bias,
        'margin': margin,
        'distinction': 'run',
        'train': 'Accuracy',
        'test': 'Accuracy',
        'train_time': 'seconds',
        'models_runs': []
    }
    train_data, rotation_matrix = concentric_spheres.get_dataloader(**experiment) 
    test_data = concentric_spheres.get_dataloader(**experiment, rotation_matrix=rotation_matrix)
    model = TwoLayerNeuralNet(**experiment).to(device)
    train_loss, test_loss = torch.nn.BCEWithLogitsLoss(), accuracy
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    run = {
        'distinction': '1',
        'train': [test(train_data, model, test_loss, device)],
        'train_time': [0],
        'test': [test(test_data, model, test_loss, device, verbose=verbose)]
    }
    experiment['models_runs'].append(run)
    summary_callback = functools.partial(gradients_summary, summary=run)
    summary_callback(model)

    for epoch in range(1, epochs + 1):
        if verbose: print(f'Epoch {epoch}')
        start_time = time.time()
        train(train_data, model, train_loss, optimizer, device, verbose=verbose, callback=summary_callback)
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
            if plot_results_on_canvas: plot_experiment(experiment, plot_results_on_canvas)
            for callback in callbacks: callback(dataloader=train_data, model=model, rotation_matrix=rotation_matrix,
                                                run=run, epoch=epoch, **experiment, **kwargs)

    if plot_results: plot_experiment(experiment)
    return experiment

def caultulate_effective_ranks(eigenvalues):
    return [sum(eigenvalues[k:]) / eigenvalue for k, eigenvalue in enumerate(eigenvalues[:-1]) if eigenvalue > 0]
    
def gradients_summary(model, *args, summary={}, epoch_frequency=100):
    epoch = len(summary.get('train', []))
    if epoch % epoch_frequency != 0: return
    outer_product_d_L_d_sigma = model.dummy_variable1.grad.detach().cpu().numpy()
    eigenvalues = sorted(numpy.abs(numpy.linalg.eigvals(outer_product_d_L_d_sigma).real).tolist(), reverse=True)
    r_k = caultulate_effective_ranks(eigenvalues)[:-1]

    summary['r_k_d_L_d_sigma'] = summary.get('r_k_d_L_d_sigma', []) + [r_k]
    summary['eigenspectrum_d_L_d_sigma'] = summary.get('eigenspectrum_d_L_d_sigma', []) + [eigenvalues]

    outer_product_sigma = model.activations.t().mm(model.activations).detach().cpu().numpy()
    eigenvalues = sorted(numpy.abs(numpy.linalg.eigvals(outer_product_sigma).real).tolist(), reverse=True)
    r_k = caultulate_effective_ranks(eigenvalues)[:-1]
    summary['r_k_sigma'] = summary.get('r_k_sigma', []) + [r_k]
    summary['eigenspectrum_sigma'] = summary.get('eigenspectrum_sigma', []) + [eigenvalues]


class TwoLayerNeuralNet(torch.nn.Module):

    def __init__(self, input_dimension:int, hidden_units:int, initialization_variance:float=1., bias:bool=True, *args, **kwargs):
        super(TwoLayerNeuralNet, self).__init__()
        self.device = 'cpu'
        self.hidden_units = hidden_units
        self.input_dimension = input_dimension
        if hidden_units % 2 == 1:
            hidden_units -= 1
            print(f'Only even number of hidden units allowed. Switching to hidden units = {hidden_units}') 

        self.input_layer = torch.nn.Linear(input_dimension, hidden_units, bias=bias)
        self.activation = torch.nn.ReLU()
        output_layer_weights = [1. / hidden_units ** 0.5] * (hidden_units // 2) + [-1. / hidden_units ** 0.5] * (hidden_units // 2)
        self.output_layer_weights = torch.tensor(output_layer_weights)

        self.dummy_variable1 = torch.zeros(hidden_units, hidden_units, requires_grad=True)
        self.dummy_variable1.retain_grad()

        self.dummy_variable2 = torch.zeros(input_dimension, input_dimension, requires_grad=True)
        self.dummy_variable2.retain_grad()
        
    def forward(self, x):
        self.pre_activations = self.input_layer(x).requires_grad_()
        self.pre_activations.retain_grad()
        self.activations = self.activation(self.pre_activations).requires_grad_()
        self.activations.retain_grad()
        output = torch.matmul(self.activations, self.output_layer_weights).unsqueeze(1)
        return output

    def to(self, device):
        super().to(device)
        self.output_layer_weights = self.output_layer_weights.to(device)
        self.device = device
        return self
