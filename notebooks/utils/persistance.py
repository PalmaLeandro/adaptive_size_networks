import os, json

EXPERIMENTS_FILE_EXTENSION = '.json'
EXPERIMENTS_DEFAULT_FOLDER = './experiments/'

def check_path(path):
    if not os.path.exists(path): 
        os.makedirs(path)

def is_numeric(string):
    return string.replace('.', '', 1).replace('e', '', 1).replace('-', '', 1).isdigit()

def is_boolean(string):
    return string.lower() in ('true', 'false')

def parse_parameter_value(string):
    if is_numeric(string):
        return float(string) if '.' in string else int(string)
    else:
        return True if string.lower() == 'true' else False

def parameters_from_file_path(file_path):
    file_name = file_path.split(os.path.sep)[-1]
    file_name_without_extension = '.'.join(file_name.split('.')[:-1])

    # Split name, extract the values and names, and build a dictionary by using them as keys and values
    parameters_names_and_values = list(filter(lambda x: x, file_name_without_extension.split('_')))

    values_indices = [index for index, string in enumerate(parameters_names_and_values) if is_numeric(string) or is_boolean(string)]
    parameters_values = list(filter(lambda string: is_numeric(string) or is_boolean(string), parameters_names_and_values))

    names_indices_from = [0] + [value_index + 1 for value_index in values_indices]
    names_indices_to = values_indices + [len(parameters_names_and_values)]

    parameter_names = [
        '_'.join(parameters_names_and_values[index_from: index_to]) 
        for index_from, index_to in zip(names_indices_from, names_indices_to)
    ]

    return {
        parameter_name: parse_parameter_value(parameter_value)
        for parameter_name, parameter_value in zip(parameter_names, parameters_values)
    }

def pick_parameters(parameters, name_parameters):
    return {parameter: parameters[parameter] for parameter in name_parameters}

def name_from_parameters(parameters):
    return '_'.join([f'{key}_{value}'for key, value in parameters.items()])

def non_iterable_parameters(parameters):
    return [parameter for parameter, value in parameters.items() if not isinstance(value, (list, dict, tuple))]

def file_name_from_parameters(name_parameters=None, **parameters):
    if name_parameters is None:
        name_parameters = non_iterable_parameters(parameters)

    return name_from_parameters(parameters if name_parameters is None else pick_parameters(parameters, name_parameters))

def file_path_from_parameters(parameters, name_parameters=None, prefix='', suffix=''):
    name_parameters_in_parameters = parameters.pop('name_parameters', None)
    file_name = file_name_from_parameters(name_parameters or name_parameters_in_parameters, **parameters)
    return f'{prefix}{file_name}{suffix}'

def save_experiment(experiment, name_parameters=None, path=EXPERIMENTS_DEFAULT_FOLDER): 
    check_path(path)
    name_parameters = name_parameters or experiment.get('name_parameters', None)
    if name_parameters is None:
        name_parameters = non_iterable_parameters(experiment)

    experiment_file_path = file_path_from_parameters(experiment, name_parameters, prefix=path, suffix='.json')
    with open(experiment_file_path, 'w', encoding='utf8') as fp:
        json.dump(experiment, fp, indent=2)

def load_experiment(path=EXPERIMENTS_DEFAULT_FOLDER, name_parameters=None, **parameters):
    if parameters: # path specifies a folder containing and the parameters specify the file.
        if name_parameters is None:
            name_parameters = non_iterable_parameters(parameters)

        path = file_path_from_parameters(parameters, name_parameters, prefix=path, suffix=EXPERIMENTS_FILE_EXTENSION)
    
    with open(path, 'r') as fp:
        experiment = json.load(fp)

    return experiment

def experiment_exists(path=EXPERIMENTS_DEFAULT_FOLDER, name_parameters=None, **parameters):
    if parameters: # path specifies a folder containing and the parameters specify the file.
        if name_parameters is None:
            name_parameters = non_iterable_parameters(parameters)

        path = file_path_from_parameters(parameters, name_parameters, prefix=path, suffix=EXPERIMENTS_FILE_EXTENSION)

    return os.path.exists(path)

def save_model(model, models_path):
    import torch
    
    check_path(models_path)
    model_file_path = file_path_from_parameters(model.persitance_parameters, 
                                                prefix=models_path, 
                                                suffix=model.MODEL_FILE_EXTENSION)
    torch.save(model.state_dict(), model_file_path)

def load_model(model_file_path, model_class=None, device=None, **parameters):
    import torch

    if model_class is not None:
        model = model_class(**parameters)  # Recommended way to load a model according to PyTorch tutorial.
        model.load_state_dict(torch.load(model_file_path))

    else:
        model = torch.load(model_file_path)

    return model.to(device) if device is not None else model


class PersistableModel(object):

    MODEL_NAME_PARAMETERS = []
    MODEL_FILE_EXTENSION = '.pt'
    MODELS_DEFAULT_FOLDER = './models/'

    @property
    def persitance_parameters(self):
        return {parameter: self.__getattribute__(parameter) for parameter in self.MODEL_NAME_PARAMETERS}
    
    def save(self, path=MODELS_DEFAULT_FOLDER):
        save_model(self, path)

    @classmethod
    def load(cls, path=MODELS_DEFAULT_FOLDER, **parameters):
        if parameters:
            path = file_path_from_parameters(parameters, cls.MODEL_NAME_PARAMETERS, prefix=path, suffix=cls.MODEL_FILE_EXTENSION)
        
        return load_model(path, cls, **parameters)

    @classmethod
    def model_exists(cls, path=MODELS_DEFAULT_FOLDER, **parameters):
        if parameters: 
            path = file_path_from_parameters(parameters, cls.MODEL_NAME_PARAMETERS, prefix=path, suffix=cls.MODEL_FILE_EXTENSION)

        return os.path.exists(path)
    