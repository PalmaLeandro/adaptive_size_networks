import os, sys, time, numpy, torch

sys.path.append(os.path.abspath(os.path.join('..')))  # Allow repository modules to be imported

from utils.optimization import initialize, train, test
from utils.persistance import save_experiment, PersistableModel
from settings.two_layer_teacher_network import get_dataloader

EXPERIMENT_NAME_PARAMETERS = ['seed', 'input_dimension', 'output_dimension', 'sample_size', 'batch_size', 
                              'learning_rate', 'max_epoch', 'convergence_epsilon']

def execute_experiment(seed, input_dimension, output_dimension, sample_size, batch_size, max_epoch, learning_rate, 
                       convergence_epsilon, runs, save_models_path=None, save_experiments_path=None, 
                       plot_results=False, plot_results_on_canvas=None, saving_epochs_interval=1, verbose=False):
    device = initialize(seed)
    train_loader, true_output_weights = get_dataloader(input_dimension, sample_size, batch_size)
    test_loader = get_dataloader(input_dimension, sample_size, batch_size, true_output_weights)
    experiment = {
        'seed': seed,
        'input_dimension': input_dimension,
        'output_dimension': output_dimension,
        'sample_size': sample_size,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'max_epoch': max_epoch,
        'convergence_epsilon': convergence_epsilon,
        'distinction': 'run',
        'true_output_weights': true_output_weights.tolist(),
        'train': 'Mean Squared Error',
        'test': 'Mean Squared Error',
        'train_time': 'seconds',
        'models_runs': []
    }
    if plot_results or plot_results_on_canvas: from utils.plotting import plot_experiment

    for run_number in range(runs):
        if verbose: print(f'Run {run_number}')

        model = SplittingExperimentTwoLayerNeuralNet(**experiment, device=device, run=run_number).to(device)
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        run = {
            'distinction': run_number,
            'train': [test(train_loader, model, loss_fn, device)],
            'test': [test(test_loader, model, loss_fn, device, verbose=verbose)],
            'train_time': [0], 
            'num_neurons': [model.num_neurons]
        }
        experiment['models_runs'].append(run)

        epoch = 1
        while epoch <= max_epoch:
            if verbose: print(f'Epoch {epoch}')

            start_time = time.time()
            train_loss = train(train_loader, model, loss_fn, optimizer, device, verbose=verbose)
            end_time = time.time()
            train_time = run['train_time'][-1] + end_time - start_time
            test_loss = test(test_loader, model, loss_fn, device, verbose=verbose)

            if 0 <= min(run['train']) - train_loss < convergence_epsilon:
                if verbose: print(f'Convergence achieve according to convergence_epsilon = {convergence_epsilon}.')

                start_time = time.time()
                model.split()
                end_time = time.time()
                train_time += end_time - start_time

                if run['num_neurons'][-1] < model.num_neurons:
                    # The optimizer needs to be recreated to account for the new neurons' parameters.
                    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
                    if verbose: print(f'Model increased size to {model.num_neurons} neurons.\n')
                else: 
                    if verbose: print(f'All eigenvalues are positive, no splitting is possible.\n')

            run['train'].append(train_loss); run['test'].append(test_loss)
            run['train_time'].append(train_time); run['num_neurons'].append(model.num_neurons)
            if (epoch % saving_epochs_interval == 0 or epoch == max_epoch):
                if save_models_path: model.save(save_models_path)
                if save_experiments_path: save_experiment(experiment, save_experiments_path, EXPERIMENT_NAME_PARAMETERS)

            if plot_results_on_canvas: plot_experiment(experiment, plot_results_on_canvas, second_axis='num_neurons')

            epoch += 1
                
    if plot_results: plot_experiment(experiment, second_axis='num_neurons')
    return experiment

    
class SplittingExperimentTwoLayerNeuralNet(torch.nn.Module, PersistableModel):
    """
    https://arxiv.org/pdf/1910.02366.pdf
    """
    MODEL_NAME_PARAMETERS = EXPERIMENT_NAME_PARAMETERS + ['run']                  

    def __init__(self, input_dimension:int, output_dimension:int, num_neurons:int=1, beta:float=3., *args, **kwargs):
        super(SplittingExperimentTwoLayerNeuralNet, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.num_neurons = num_neurons
        self.beta = beta
        self.device = None

        self.linear1 = torch.nn.Linear(self.input_dimension, self.num_neurons, bias=False)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(self.num_neurons, self.output_dimension, bias=False)
        
        self.dummy_variable = torch.zeros(self.input_dimension, self.input_dimension, self.num_neurons, requires_grad=True)
        self.dummy_variable.retain_grad()

        self.store_parameters(input_dimension=input_dimension,
                              output_dimension=output_dimension,
                              num_neurons=num_neurons,
                              beta=beta,
                              **kwargs)

    def forward(self, x):
        pre_activation = self.linear1(x)
        activations = self.activation(pre_activation)
        output = self.linear2(activations + self.splitting_matrix_calculation(x))
        return output

    def to(self, device):
        super().to(device)
        self.dummy_variable = self.dummy_variable.to(device)
        self.dummy_variable.retain_grad()
        self.device = device
        return self

    def splitting_matrix_calculation(self, X):
        """ This method adds a dummy variable which is always zero to the forward pass. Adding it doesn't change 
        the calculation of the neural net but allows to calculate the splitting matrix via auto-differentiation.
    
        The size of the dummy variable is the number of neurons times the neurons' input dimesion, d, squared
        as we need one splitting matrix of size d by d for each neuron to evaluate if split it or not.
    
        Considering the rest of the neural net fixed, the splitting matrix S_j of neuron j called \sigma_j is 
        
        S_j(\theta_j) = (1/n) sum_i^N [ \nabla_\sigma_j L(\sigma_j(x), y) \nabla^2_\theta_j(x_i^T \theta_j) ]
    
        \nabla^2_\theta_j( x_i^T \theta_j ) = \sigma''(x_i^T \theta_j) * x_i * x_i^T
    
        where x_i is the sample i input. We introduce the dummy variable in the calculation as follows
    
        \tilde{\nabla^2_\theta_j( x_i^T \theta_j )} = \sigma''(x_i^T \theta_j) * x_i * dummy_variable * x_i^T
    
        Since the ReLU doesn't have a second derivative, \sigma'', we approximate it by the softplus. 
        The second derivative of softplus(x) is beta * s(x) * (1 - s(x)), there s is sigmoid and beta is some scaling.
        """

        neurons_pre_activation = self.linear1(X)
        s = torch.sigmoid(neurons_pre_activation * self.beta)
        relu_second_derivative_approximation = self.beta * s * (1. - s)

        # The following calculation performs X^t * dummy_variable * X and returns a tensor of shape [batch_size, num_neurons]
        input_covariance_matrix = (
            X.unsqueeze(1).bmm(
                X.matmul(self.dummy_variable).reshape(-1, self.input_dimension, self.num_neurons)
            ).reshape(-1, self.num_neurons)
        )

        return relu_second_derivative_approximation * input_covariance_matrix

        
    def split(self):
        splitting_matrices = self.dummy_variable.grad.data.reshape(
            self.num_neurons, self.input_dimension, self.input_dimension
        )
        neuron_to_split = neuron_to_split_eigenvector = None
        min_neurons_eigenvalue = 0
        for neuron, splitting_matrix in enumerate(splitting_matrices):
            splitting_matrix = splitting_matrix.data.cpu()
            eigenvalues, eigenvectors = numpy.linalg.eig(splitting_matrix)
            min_eigenvalue_index = eigenvalues.argmin()
            min_eigenvalue = eigenvalues[min_eigenvalue_index]
            min_eigenvalue_eigenvector = eigenvectors[min_eigenvalue_index]

            if min_eigenvalue < min_neurons_eigenvalue:
                min_neurons_eigenvalue = min_eigenvalue
                neuron_to_split = neuron
                neuron_to_split_eigenvector = min_eigenvalue_eigenvector

        if neuron_to_split is not None:
            split_directions = torch.Tensor(numpy.array([neuron_to_split_eigenvector]))
            if self.device is not None: split_directions = split_directions.to(self.device)
            self.split_input_weights([neuron_to_split], split_directions)
            self.split_output_weights([neuron_to_split])
            self.num_neurons += 1
            self.dummy_variable = torch.zeros(self.input_dimension, self.input_dimension, self.num_neurons, requires_grad=True)
            self.dummy_variable.retain_grad()
            if self.device is not None: self.to(self.device)


    def split_input_weights(self, neurons_to_split, splitting_directions, splitting_epsilon=0.01):
        """ The splitting of the input weights is done by multiplying the original neuron input weights
        by epsilon times the splitting direction and adding additional neurons with the original 
        neuron input weights multiplied by - epsilon times the splitting direction.
        """

        input_layer_weights = self.linear1.weight.data
        new_neurons_input_weights = input_layer_weights[neurons_to_split].clone() - splitting_epsilon * splitting_directions
        input_layer_weights[neurons_to_split] += splitting_epsilon * splitting_directions
        new_input_layer_weights = torch.cat([input_layer_weights, new_neurons_input_weights])
        self.linear1 = torch.nn.Linear(self.input_dimension, self.num_neurons + len(neurons_to_split), bias=False)
        with torch.no_grad():
            self.linear1.weight.copy_(new_input_layer_weights)


    def split_output_weights(self, neurons_to_split):
        """ The splitting of the output weights is done by multiplying the original neuron output weights
        by epsilon times the splitting direction and adding additional neurons with the original 
        neuron output weights multiplied by - epsilon times the splitting direction.
        """

        output_layer_weights = self.linear2.weight.data[0]
        new_neurons_output_weights = 0.5 * output_layer_weights[neurons_to_split].clone()
        output_layer_weights[neurons_to_split] *= 0.5
        new_output_layer_weights = (
            torch.cat([output_layer_weights, new_neurons_output_weights])
            .reshape(self.output_dimension, self.num_neurons + len(neurons_to_split))
        )
        self.linear2 = torch.nn.Linear(self.num_neurons + len(neurons_to_split), self.output_dimension, bias=False)
        with torch.no_grad():
            self.linear2.weight.copy_(new_output_layer_weights)
