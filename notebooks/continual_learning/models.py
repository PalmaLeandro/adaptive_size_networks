import scipy, torch, einops

class Neuron(torch.nn.Module):

    def __init__(self, input_dimension, output_dimension, initialization_scale=1., *args, **kwargs):
        super(Neuron, self).__init__()
        self.input_layer = torch.nn.Linear(in_features=input_dimension, out_features=1, bias=False)
        self.output_layer = torch.nn.Linear(in_features=1, out_features=1 if output_dimension <= 2 else output_dimension, bias=False)
        self.frozen = False
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
        for parameter in self.parameters(): 
            parameter.requires_grad = False

        self.frozen = True

    @property
    def norm(self):
        return self.output_layer.weight.norm() * self.input_layer.weight.norm()

class Growing2LayerReLUNN(torch.nn.Module):

    def __init__(self, seed, input_dimension, backbone, new_classes_per_iteration, new_neurons_per_iteration, 
                 device='cpu', epoch=0, initialization_scale=1., classes=None, *args, **kwargs):
        super(Growing2LayerReLUNN, self).__init__()
        self.seed = seed
        self.input_dimension = input_dimension
        self.device = device
        self.epoch = epoch
        self.initialization_scale = initialization_scale
        self.backbone = backbone
        self.n_observed_classes = new_classes_per_iteration
        self.new_classes_per_iteration = new_classes_per_iteration
        self.new_neurons_per_iteration = new_neurons_per_iteration
        output_dimension = min(self.new_classes_per_iteration, classes or self.new_classes_per_iteration)
        output_dimension = output_dimension if output_dimension > 2 else 1
        self.neurons = torch.nn.ModuleList([Neuron(self.input_dimension, output_dimension, self.initialization_scale) 
                                            for _ in range(self.new_neurons_per_iteration)])
        self.layers = [self]

    def forward(self, x):
        x = self.backbone(x).to(self.device)
        self.activations = torch.stack([neuron(x) for neuron in self.neurons]).squeeze()
        self.output = self.activations.sum(dim=0).requires_grad_()
        self.output.retain_grad()
        return self.output
 
    def increase_width(self):
        #previously_observed_classes = self.n_observed_classes
        self.n_observed_classes += self.new_classes_per_iteration
        
        for neuron in self.neurons:
            neuron.increase_output_dimension(self.n_observed_classes)
            neuron.freeze()
        
        for _ in range(self.new_neurons_per_iteration):
            self.neurons.append(Neuron(self.input_dimension, self.n_observed_classes, self.initialization_scale))

    @property
    def norm(self):
        return sum([neuron.norm for neuron in self.neurons])

    @property
    def weight(self):
        return torch.cat([neuron.weight for neuron in self.neurons])

    def prune(self, inputs, labels):
        self(inputs)
        outputs = []
        neurons_norms = []
        frozen_neurons_outputs = []
        for neuron in self.neurons:
            if neuron.frozen:
                frozen_neurons_outputs.append(neuron.activations)

            else:
                outputs.append(neuron.activations)
                neurons_norms.append(neuron.norm)
        
        outputs = torch.stack(outputs)
        neurons_norms = torch.tensor(neurons_norms)
        labels_ = labels
        if frozen_neurons_outputs:
            frozen_neurons_outputs = torch.stack(frozen_neurons_outputs).sum(dim=0)
            labels_ -= frozen_neurons_outputs

        labels_ = einops.repeat(labels_, 'n c -> k n c', k=len(neurons_norms))
        gamma = (labels_ * ((outputs.squeeze() / neurons_norms.reshape(-1, 1, 1)))).sum(dim=-1).T

        initial_solution = neurons_norms.detach().cpu().numpy()
        A = - gamma.detach().cpu().numpy()
        b = - torch.ones(len(inputs)).numpy()
        c = torch.ones(len(initial_solution))
        final_solution = scipy.optimize.linprog(c, x0=initial_solution, A_ub=A, b_ub=b).x
                
        new_list_of_neurons = torch.nn.ModuleList([])
        for neuron, alpha in zip(self.neurons, final_solution):
            if neuron.frozen:
                new_list_of_neurons.append(neuron)

            if alpha > 0:
                new_neuron = Neuron(self.input_dimension, output_dimension=self.n_observed_classes)
                with torch.no_grad():
                    new_input_layer_weights = (neuron.input_layer.weight / neuron.input_layer.weight.norm()) * alpha ** .5
                    new_output_layer_weights = (neuron.output_layer.weight / neuron.output_layer.weight.norm()) * alpha ** .5
                    new_neuron.input_layer.weight.copy_(new_input_layer_weights)
                    new_neuron.output_layer.weight.copy_(new_output_layer_weights)

                new_list_of_neurons.append(new_neuron)

        del self.neurons
        self.neurons = new_list_of_neurons
