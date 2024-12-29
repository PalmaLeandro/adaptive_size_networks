import numpy, scipy, torch
from .persistance import PersistableModel


class FullyConnectedNeuralNetwork(torch.nn.Module, PersistableModel):

    MODEL_NAME_PARAMETERS = ['epoch']

    def __init__(self, input_dimension:int, target_accuracy:float=1., initial_hidden_units:int=1, 
                 initial_depth:int=2, bias:bool=True, architecture=None, classes:int=2, epoch:int=0, 
                 device='cpu', seed='not_specified', balanced_initialization=False, 
                 initialization_scale:float=1, initial_weights=None, initial_biases=None, 
                 output_layer_initial_weights=None, negative_slope:float=0., *args, **kwargs):
        super(FullyConnectedNeuralNetwork, self).__init__()
        self.input_dimension = input_dimension
        self.target_accuracy = target_accuracy
        self.initial_hidden_units = initial_hidden_units
        self.initial_depth = initial_depth
        self.classes = classes
        self.bias = bias
        self.epoch = epoch
        self.device = device
        self.seed = seed
        self.initialization_scale = initialization_scale
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(input_dimension if layer_index == 0 else hiden_units, hiden_units, bias=bias) 
            for layer_index, hiden_units in enumerate(architecture or [initial_hidden_units,] * (initial_depth - 1))
        ]).to(device)
        self.activation_function = torch.nn.LeakyReLU(negative_slope=negative_slope).to(device)
        output_units = self.layers[-1].weight.shape[0]
        self.output_layer = torch.nn.Linear(output_units, 1 if self.classes == 2 else self.classes, bias=False).to(device)
        with torch.no_grad(): 
            for layer_index, layer in enumerate(self.layers):
                if initial_weights is None:
                    torch.nn.init.normal_(layer.weight, std=initialization_scale ** 0.5)
                else:
                    layer.weight.copy_(torch.tensor(initial_weights[layer_index]).to(device))

                if bias:
                    if initial_biases is None:
                        layer.bias.copy_(torch.zeros(layer.bias.shape[0]).to(device))
                    else:
                        layer.bias.copy_(torch.tensor(initial_biases[layer_index]).to(device))
                        
            if output_layer_initial_weights is not None:
                self.output_layer.weight.copy_(torch.tensor(output_layer_initial_weights).to(device))
            
            elif balanced_initialization:
                self.output_layer.weight.copy_(self.output_layer.weight.sign() * torch.tensor(self.layers[-1].weight.norm(dim=1)))
        
    @property        
    def depth(self):
        return len(self.layers) + 1

    def parameters(self):
        parameters = list(self.output_layer.parameters())
        for layer in self.layers: parameters += list(layer.parameters())
        return parameters
        
    def forward(self, x):
        x = x.requires_grad_()
        x.retain_grad()
        self.pre_activations = []
        self.activations = [x]
        for layer in self.layers:
            pre_activation = layer(x).requires_grad_()
            pre_activation.retain_grad()
            self.pre_activations.append(pre_activation)
            activation = self.activation_function(pre_activation).requires_grad_()
            activation.retain_grad()
            self.activations.append(activation)
            x = activation
            
        self.output = torch.matmul(activation, self.output_layer.weight.t()).squeeze().requires_grad_()
        self.output.retain_grad()
        return self.output

    def to(self, device):
        super().to(device)
        for layer_index, layer in enumerate(self.layers):
            self.layers[layer_index] = layer.to(device)

        self.output_layer = self.output_layer.to(device)
        self.device = device 
        return self
    
    @property
    def architecture(self):
        return [layer.weight.shape[0] for layer in self.layers]
    
    @property
    def metrics(self):
        self.active_samples_count = []
        self.positive_margins_count = []
        self.samples_not_captured_count = []
        self.dead_units = []
        metrics = []
        for layer_index, (layer, inputs, pre_activation) in enumerate(zip(self.layers, self.activations[:-1], self.pre_activations)):
            pre_activation_gradient_average_norm = pre_activation.grad.abs().mean(dim=0).tolist()
            pre_activation_average_gradient_norm = pre_activation.grad.mean(dim=0).abs().tolist()
            per_sample_gradients = pre_activation.grad.unsqueeze(-1).bmm(inputs.unsqueeze(1))
            dead_units = (per_sample_gradients.norm(dim=0).norm(dim=1) == 0).nonzero().reshape(-1).detach().cpu().tolist()
            margins = - per_sample_gradients.permute(1, 0, 2).bmm(torch.div(layer.weight, layer.weight.norm(dim=1).unsqueeze(1)).unsqueeze(2)).squeeze(2)
            active_samples_count = (per_sample_gradients.norm(dim=2) != 0).count_nonzero(dim=0).detach().cpu().tolist()
            samples_not_captured_count = (per_sample_gradients.norm(dim=2).norm(dim=1) == 0).count_nonzero(dim=0).item()
            positive_margins_count = (margins > 0).count_nonzero(dim=1).detach().cpu().tolist()
            margins = margins.detach().cpu().tolist()
            gradients_average_norm = per_sample_gradients.norm(dim=2).mean(dim=0).tolist()
            average_gradient_norm = per_sample_gradients.mean(dim=0).norm(dim=1).tolist()

            self.active_samples_count.append(active_samples_count)
            self.samples_not_captured_count.append(samples_not_captured_count)
            self.positive_margins_count.append(positive_margins_count)
            self.dead_units.append(dead_units)
            
            metrics.append({
                'epoch': self.epoch,
                'layer': layer_index,
                'active_samples_count': active_samples_count,
                'positive_margins_count': positive_margins_count,
                'samples_not_captured_count': samples_not_captured_count,
                'gradients_average_norm': gradients_average_norm,
                'average_gradient_norm': average_gradient_norm,
                'pre_activation_gradient_average_norm': pre_activation_gradient_average_norm,
                'pre_activation_average_gradient_norm': pre_activation_average_gradient_norm,
                'margins': margins,
                'dead_units': dead_units
            })

        return metrics
    
    @property
    def norms(self):
        weights_products = torch.norm(self.layers[0].weight, dim=1)
        for layer in self.layers[1:]:
            weights_products = torch.norm(layer.weight) @ weights_products

        return (weights_products.T * self.output_layer.weight.norm(dim=0)).squeeze()
    
    @property
    def norm(self):
        return self.weights_products.sum()
    
    @property
    def input_layer(self):
        return self.layers[0]
    
    def prune(self, labels):
        labels = ((labels * 2.) - 1.)
        outputs = self.activations[-1] * self.output_layer.weight
        gamma = torch.diag((labels * 2. - 1.).flatten()) @ (outputs.squeeze() / self.weights_products.unsqueeze(1)).T

        initial_solution = self.weights_products.detach().cpu().numpy()
        A = - gamma.detach().cpu().numpy()
        b = - torch.ones(len(labels)).numpy()
        c = torch.ones(len(initial_solution))
        final_solution = scipy.optimize.linprog(c, x0=initial_solution, A_ub=A, b_ub=b).x
        
        non_zero_neurons = final_solution.nonzero()[0]
        new_input_layer = torch.nn.Linear(self.input_dimension, len(non_zero_neurons), self.bias)
        new_output_layer = torch.nn.Linear(len(non_zero_neurons), 1 if self.classes <= 2 else self.classes, self.bias)
        with torch.no_grad():
            old_input_layer_normalized = self.layer[0].weight / self.layer[0].weight.norm(dim=0)
            old_output_layer_normalized = self.output_layer.weight / self.output_layer.weight.norm(dim=0)
            new_input_layer.weight.copy_(old_input_layer_normalized[non_zero_neurons] * final_solution[non_zero_neurons] ** 0.5)
            new_output_layer.weight.copy_(old_output_layer_normalized[non_zero_neurons] * final_solution[non_zero_neurons] ** 0.5)
            self.layers[0] = new_input_layer
            self.output_layer = new_output_layer
    
class GrowingFullyConnectedNeuralNetwork(FullyConnectedNeuralNetwork):
    
    def reinitialize_from_layer(self, layer_index):
        input_dimension = self.input_dimension if layer_index == 0 else self.layers[layer_index - 1].weight.shape[0]
        self.layers[layer_index] = torch.nn.Linear(
            input_dimension if layer_index == 0 else self.initial_hidden_units, self.initial_hidden_units, 
            bias=self.bias
        ).to(self.device)
        for _ in range(layer_index + 1, len(self.layers)): 
            del self.layers[layer_index + 1]

        output_units = self.layers[-1].weight.shape[0]
        self.output_layer = torch.nn.Linear(output_units, 1 if self.classes == 2 else self.classes, bias=False).to(self.device)

    def prune(self):
        has_removed_units = False
        for layer_index, dead_units in enumerate(self.dead_units):
            hidden_units = self.layers[layer_index].weight.shape[0]
            if dead_units:
                if len(dead_units) == hidden_units:
                    self.reinitialize_from_layer(layer_index)
                    print(f'All {len(dead_units)} units at layer {layer_index} are dead. The network will be reinitialized from layer {layer_index} to be of depth {self.depth}') 
                    self.report_architecture_change('layer_removed')
                    return True
                
                else:
                    self.remove_dead_units_from_layer(dead_units, layer_index)
                    has_removed_units = True

        return 'units_removed' if has_removed_units else None
    
    def remove_dead_units_from_layer(self, dead_units, layer_index):
        input_dimension = self.input_dimension if layer_index == 0 else self.layers[layer_index - 1].weight.shape[0]
        number_of_units = self.layers[layer_index].weight.shape[0]
        alive_units = [unit_index for unit_index in range(number_of_units) if unit_index not in dead_units]
        alive_units_weights = self.layers[layer_index].weight.data[alive_units]
        alive_units_biases = self.layers[layer_index].bias.data[alive_units]
        self.layers[layer_index] = torch.nn.Linear(input_dimension, len(alive_units), bias=self.bias).to(self.device)
        with torch.no_grad():
            self.layers[layer_index].weight.copy_(alive_units_weights)
            self.layers[layer_index].bias.copy_(alive_units_biases)

        if layer_index == self.depth - 2: 
            output_layer_weights = self.output_layer.weight[:, alive_units].clone()
            output_units = output_layer_weights.shape[1]
            self.output_layer = torch.nn.Linear(output_units, 1 if self.classes == 2 else self.classes, bias=False).to(self.device)
            with torch.no_grad(): 
                self.output_layer.weight.copy_(output_layer_weights)

        else:
            next_layer_units = self.layers[layer_index + 1].weight.shape[0]
            next_layer_alive_units_weights = self.layers[layer_index + 1].weight[:, alive_units].clone()
            next_layer_alive_units_biases = self.layers[layer_index + 1].bias.data.clone()
            self.layers[layer_index + 1] = torch.nn.Linear(len(alive_units), next_layer_units, bias=self.bias).to(self.device)
            with torch.no_grad():
                self.layers[layer_index + 1].weight.copy_(next_layer_alive_units_weights)
                self.layers[layer_index + 1].bias.copy_(next_layer_alive_units_biases)

        print(f'{len(dead_units)} units at layer {layer_index} are dead and were removed')
    
    def grow(self):
        for layer_index, (active_samples_count, samples_not_captured_count, positive_margins_count) in enumerate(zip(self.active_samples_count, 
                                                                                                         self.samples_not_captured_count, 
                                                                                                         self.positive_margins_count)):
            if samples_not_captured_count:
                self.grow_width(layer_index)
                return 'units_added'
            
            if (torch.tensor(positive_margins_count) < torch.tensor(active_samples_count) * self.target_accuracy).count_nonzero().squeeze().item():
                self.grow_depth(layer_index)
                return 'layer_added'

        return None
                    
    def grow_width(self, layer_index):
        former_weights = self.layers[layer_index].weight.data.clone()
        former_biases = self.layers[layer_index].bias.data.clone()
        input_dimension = former_weights.shape[1]
        hidden_units = former_weights.shape[0]
        self.layers[layer_index] = torch.nn.Linear(input_dimension, hidden_units + 2, bias=self.bias).to(self.device)
        with torch.no_grad():
            self.layers[layer_index].weight[:-2].copy_(former_weights)
            self.layers[layer_index].bias[:-2].copy_(former_biases)
            self.layers[layer_index].bias[-2:].copy_(torch.tensor([0.,] * 2).to(self.device))

        if layer_index == self.depth - 2:    
            output_units = self.layers[-1].weight.shape[0]
            former_output_layer_weights = self.output_layer.weight.clone()
            self.output_layer = torch.nn.Linear(output_units, 1 if self.classes == 2 else self.classes, bias=False).to(self.device)
            with torch.no_grad(): 
                self.output_layer.weight[:, :-2].copy_(former_output_layer_weights)
        
        else:
            next_layer_former_weights = self.layers[layer_index + 1].weight.data.clone()
            next_layer_former_biases = self.layers[layer_index + 1].bias.data.clone()
            next_layer_hidden_units = self.layers[layer_index + 1].weight.shape[0]
            self.layers[layer_index + 1] = torch.nn.Linear(hidden_units + 2, next_layer_hidden_units, bias=self.bias).to(self.device)
            with torch.no_grad():
                self.layers[layer_index + 1].weight[:, :-2].copy_(next_layer_former_weights)
                self.layers[layer_index + 1].bias.copy_(next_layer_former_biases)
        
        print(f'Width growth: Two unit with opposing signs were added to layer {layer_index} which now has {hidden_units + 2} units')

    def grow_depth(self, layer_index):
        hidden_units = self.layers[layer_index].weight.shape[0]
        self.layers.insert(layer_index + 1, torch.nn.Linear(hidden_units, hidden_units, self.bias).to(self.device))
        with torch.no_grad():
            self.layers[layer_index + 1].weight.copy_(torch.eye(hidden_units).to(self.device))
            self.layers[layer_index + 1].bias.copy_(torch.zeros(hidden_units).to(self.device))
            
        print(f'Depth growth: A ReLU layer with identity weights was inserted at depth {layer_index + 1}. Total depth including output layer is {self.depth}')


class EpsilonNetFullyConnectedNeuralNetwork(FullyConnectedNeuralNetwork):

    def __init__(self, input_dimension:int, net_epsilon:float, initialization_scale:float, overparametrization:int=1, 
                 repeat_nodes=True, *args, **kwargs):
        angle = 2 * numpy.arcsin(net_epsilon / (2 * initialization_scale)) # chord formula
        number_of_nodes = int(2 * numpy.pi / (2 * angle)) + 1
        nodes = []
        for node_index in range(number_of_nodes):
            node = initialization_scale * numpy.array([numpy.sin(2 * numpy.pi * (node_index / number_of_nodes)), 
                                                       numpy.cos(2 * numpy.pi * (node_index / number_of_nodes))])
            if input_dimension > 2:
                node = numpy.concatenate([node, numpy.zeros(input_dimension - 2)])

            nodes.append(node)

        input_layer_weights = numpy.repeat(nodes, int((2 if repeat_nodes else 1) * overparametrization), axis=0)

        output_layer_weights = initialization_scale * numpy.tile([-1, 1], int(len(nodes) / ((1 if repeat_nodes else 2)) * overparametrization))
        model_args = {
            'input_dimension': input_dimension, 
            'initial_hidden_units': len(nodes) * int(2 if repeat_nodes else 1 * overparametrization), 
            'initial_depth': 2, 
            'bias': False,
            'initial_weights': [input_layer_weights],
            'output_layer_initial_weights': output_layer_weights
        }
        super().__init__(*args, **{**kwargs, **model_args})
