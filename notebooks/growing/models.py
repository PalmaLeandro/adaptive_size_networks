import numpy, scipy, torch

class Neuron(torch.nn.Module):

    def __init__(self, input_dimension, output_dimension, initialization_scale=1., *args, **kwargs):
        super(Neuron, self).__init__()
        self.input_layer = torch.nn.Linear(in_features=input_dimension, out_features=1, bias=False)
        self.output_layer = torch.nn.Linear(in_features=1, out_features=output_dimension, bias=False)
        with torch.no_grad():
            torch.nn.init.normal_(self.input_layer.weight, std=initialization_scale ** 0.5)

    @property
    def weight(self):
        return self.input_layer.weight

    def forward(self, x):
        x = self.input_layer(x)
        x = torch.relu(x)
        self.activations = self.output_layer(x)
        return self.activations
    
    def increase_output_dimension(self, new_output_dimension):
        previous_output_layer = self.output_layer.weight.data.clone()
        self.output_layer = torch.nn.Linear(in_features=1, out_features=new_output_dimension, bias=False)
        with torch.no_grad():
            self.output_layer.weight[:previous_output_layer.shape[0], :].copy_(previous_output_layer)

    def freeze(self):
        for parameter in self.parameters(): parameter.requires_grad = False

class Growing2LayerReLUNN(torch.nn.Module):

    def __init__(self, seed, input_dimension, backbone, new_neurons_per_iteration, 
                 device='cpu', epoch=0, initialization_scale=1., classes=None, *args, **kwargs):
        super(Growing2LayerReLUNN, self).__init__()
        self.seed = seed
        self.input_dimension = input_dimension
        self.device = device
        self.epoch = epoch
        self.initialization_scale = initialization_scale
        self.backbone = backbone
        self.new_neurons_per_iteration = new_neurons_per_iteration
        self.output_dimension = classes if classes > 2 else 1
        self.neurons = torch.nn.ModuleList([Neuron(self.input_dimension, self.output_dimension, self.initialization_scale) 
                                            for _ in range(self.new_neurons_per_iteration)])
        self.layers = [self]

    def forward(self, x):
        x = self.backbone(x).to(self.device)
        x = torch.stack([neuron(x) for neuron in self.neurons]).sum(dim=0)
        return x.squeeze(1)

    @property
    def weight(self):
        return torch.cat([neuron.weight for neuron in self.neurons])

    @property
    def activations(self):
        return torch.cat([neuron.activations for neuron in self.neurons])
 
    def increase_width(self):
        for neuron in self.neurons:
            neuron.freeze()
        
        for _ in range(self.new_neurons_per_iteration):
            self.neurons.append(Neuron(self.input_dimension, self.output_dimension, self.initialization_scale))


    def prune(self, inputs, labels):
        self(inputs)
        neurons_to_remove = []
        activations = []
        for neuron in self.neurons:
            if neuron.activations.count_nonzero():
                activations.append(neuron.activations)
            
            else:
                neurons_to_remove.append(neuron)

        outputs = torch.stack(activations)
        #labels = (labels * 2. - 1).tile(len(outputs)).reshape(outputs.shape)
        #gamma = (labels * outputs).squeeze().reshape(len(inputs), len(outputs))
        gamma = (torch.diag((labels * 2. - 1.).flatten()) @ outputs.squeeze().T)
        gamma_bar = gamma / gamma.sum(dim=1).min()

        initial_solution = torch.ones(len(outputs)).numpy()
        A = - gamma_bar.detach().cpu().numpy()
        b = - torch.ones(len(inputs)).numpy()
        final_solution = scipy.optimize.linprog(initial_solution, A_ub=A, b_ub=b).x

        nonzero_neurons_indices = numpy.nonzero(final_solution)[0]
        remaining_neurons = [neuron for neuron_index, neuron in enumerate(self.neurons) if neuron_index in nonzero_neurons_indices]
        for _ in range(len(remaining_neurons)):
            for neuron_index, neuron in enumerate(self.neurons):
                if neuron not in remaining_neurons:
                    del self.neurons[neuron_index]
                    break
