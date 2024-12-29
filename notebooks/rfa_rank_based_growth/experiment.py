import os, sys, time, functools, numpy, torch

sys.path.append(os.path.abspath(os.path.join('..')))  # Allow repository modules to be imported

from utils.optimization import initialize, test, accuracy
from utils.persistance import save_experiment, PersistableModel
from settings.noisy_xor import get_dataloader

EXPERIMENT_PARAMETERS = ['seed', 'input_dimension', 'sample_size', 'batch_size', 'epochs', 'learning_rate', 'clusters_per_class',
                         'noise_rate', 'within_cluster_variance', 'hidden_units', 'initialization_variance']

def execute_experiment(seed, noise_rate, within_cluster_variance, input_dimension, sample_size, batch_size, epochs, 
                       learning_rate, hidden_units, initialization_variance, runs_per_model, clusters_per_class,
                       convergence_epsilon, save_models_path=None, save_experiments_path=None, plot_results=False, 
                       plot_results_on_canvas=None, saving_epochs_interval=1, verbose=False, callbacks=[], **kwargs):
    device = initialize(seed)
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
        'clusters_per_class': clusters_per_class,
        'distinction': 'run',
        'train': 'Accuracy',
        'test': 'Accuracy',
        'train_time': 'seconds',
        'models_runs': []
    }
    train_data, rotation_matrix = get_dataloader(**experiment)
    test_data = get_dataloader(**experiment, rotation_matrix=rotation_matrix)
    if plot_results or plot_results_on_canvas: from utils.plotting import plot_experiment

    for run_number in range(runs_per_model):
        if verbose: print(f'Run {run_number}')

        model = TwoLayerNeuralNet(**experiment, rotation_matrix=rotation_matrix, run=run_number).to(device)
        train_loss, test_loss = torch.nn.BCEWithLogitsLoss(), accuracy
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        run = {
            'distinction': run_number,
            'train': [test(train_data, model, test_loss, device)],
            'train_time': [0],
            'test': [test(test_data, model, test_loss, device, verbose=verbose)]
        }
        experiment['models_runs'].append(run)
        for epoch in range(1, epochs + 1):
            if verbose: print(f'Epoch {epoch}')
                
            start_time = time.time()
            summary_callback = functools.partial(gradients_summary, summary=run)
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

def train(dataloader, model, loss_fn, optimizer, device, verbose=False, callback=None):
    num_batches = len(dataloader)
    model.train()
    train_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        Xy = torch.concatenate([X, y], dim=1)
        loss = loss_fn(model(Xy), y)
        loss.backward()
        optimizer.step()
        if callback: callback(model)
        optimizer.zero_grad()
        train_loss += loss.item()

    train_loss /= num_batches
    if verbose: print(f"Train Avg loss: {train_loss:>8f}")
    return train_loss

def test(dataloader, model, loss_fn, device, verbose=False):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        test_loss = 0
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            Xy = torch.concatenate([X, y], dim=1)
            loss = loss_fn(model(Xy), y)
            test_loss += loss.item()

    test_loss /= num_batches
    if verbose: print(f"Test Avg loss: {test_loss:>8f}\n")    
    return test_loss
    
def gradients_summary(model, summary={}, epoch_frequency=100):
    epoch = len(summary.get('train', []))
    if epoch % epoch_frequency != 0: return
    eigenvalues = numpy.abs(numpy.linalg.eigvals(model.dummy_variable1.grad.detach().cpu().numpy()))
    eigenvalues[::-1].sort()
    rs = [float(sum(eigenvalues[k:]) / eigenvalues[k]) for k in range(min(len(eigenvalues) - 1, 9))]
    summary['activations_rs1'] = summary.get('activations_rs1', []) + [rs]

    eigenvalues = numpy.abs(numpy.linalg.eigvals(model.activations.t().mm(model.activations).detach().cpu().numpy()))
    eigenvalues[::-1].sort()
    rs = [float(sum(eigenvalues[k:]) / eigenvalues[k]) for k in range(min(len(eigenvalues) - 1, 9))]
    summary['activations_rs2'] = summary.get('activations_rs2', []) + [rs]


class TwoLayerNeuralNet(torch.nn.Module, PersistableModel):

    MODEL_NAME_PARAMETERS = EXPERIMENT_PARAMETERS + ['run']

    def __init__(self, input_dimension:int, hidden_units:int, initialization_variance:float, *args, **kwargs):
        super(TwoLayerNeuralNet, self).__init__()
        self.device = 'cpu'
        self.hidden_units = hidden_units
        self.input_dimension = input_dimension
        if hidden_units % 2 == 1:
            hidden_units -= 1
            print(f'Only even number of hidden units allowed. Switching to hidden units = {hidden_units}') 

        self.input_layer = torch.nn.Linear(input_dimension, hidden_units, bias=False)
        torch.nn.init.normal_(self.input_layer.weight, std=initialization_variance ** 0.5)

        self.activation = torch.nn.ReLU()
        self.output_layer_weights = torch.tensor([1. / hidden_units ** 0.5] * (hidden_units // 2) + [-1. / hidden_units ** 0.5] * (hidden_units // 2), requires_grad=True)
        self.output_layer_weights.retain_grad()

        #with torch.no_grad():
        #    positive_neurons = torch.sign(self.output_layer_weights) > 0
        #    negative_neurons = ~ positive_neurons
        #    self.input_layer.weight.data[:, 0] = torch.where(negative_neurons, self.input_layer.weight.data[:, 0].abs() * -1., self.input_layer.weight.data[:, 0])
        #    #self.input_layer.weight.data[:, 1] = torch.where(positive_neurons, self.input_layer.weight.data[:, 1].abs() * -1., self.input_layer.weight.data[:, 1])
        #    self.input_layer.weight.data = self.input_layer.weight.data.mm(torch.tensor(rotation_matrix, dtype=torch.float))

        self.dummy_variable1 = torch.zeros(hidden_units, hidden_units, requires_grad=True)
        self.dummy_variable1.retain_grad()

        self.dummy_variable2 = torch.zeros(hidden_units, hidden_units, requires_grad=True)
        self.dummy_variable2.retain_grad()

        self.store_parameters(input_dimension=input_dimension, 
                              hidden_units=hidden_units, 
                              initialization_variance=initialization_variance,
                              **kwargs)
        
    def forward(self, Xy):
        x = Xy[..., :-1]
        y = Xy[..., -1].unsqueeze(1) * 2. - 1.
        self.pre_activations = self.input_layer(x)
        self.activations = self.activation(self.pre_activations)
        dummy_term1 = (
            self.activations.unsqueeze(1).bmm(self.activations.mm(self.dummy_variable1)
                .reshape(-1, self.hidden_units, 1)).reshape(-1)
        )
        dummy_term2 = self.activations.mm(self.dummy_variable2).mm(self.output_layer_weights.unsqueeze(1)).squeeze()
        self.output = torch.matmul(self.activations, self.output_layer_weights) + dummy_term1 + dummy_term2
        self.output = self.output.requires_grad_()
        self.output.retain_grad()
        return self.output.unsqueeze(1)

    def to(self, device):
        super().to(device)
        self.output_layer_weights = self.output_layer_weights.to(device)
        self.output_layer_weights.retain_grad()
        self.dummy_variable1 = self.dummy_variable1.to(device)
        self.dummy_variable1.retain_grad()
        self.dummy_variable2 = self.dummy_variable2.to(device)
        self.dummy_variable2.retain_grad()
        self.device = device
        return self