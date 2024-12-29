import numpy, torch

def initialize(seed=123):
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return device, generator

def get_random_states():
    MT19937, keys, pos, has_gauss, cached_gaussian = numpy.random.get_state()
    torch_random_state = torch.get_rng_state().tolist()
    return {
        'numpy_random_state': [MT19937, keys.tolist(), pos, has_gauss, cached_gaussian],
        'torch_random_state': torch_random_state
    }

def set_random_states(random_states, **kwargs):
    MT19937, keys, pos, has_gauss, cached_gaussian = random_states['numpy_random_state']
    keys = numpy.uint32(keys)
    numpy.random.set_state((MT19937, keys, pos, has_gauss, cached_gaussian))
    torch.set_rng_state(torch.ByteTensor(random_states['torch_random_state']))
    generator = torch.Generator()
    generator.set_state(torch.ByteTensor(random_states['torch_random_state']))
    return generator

def train(dataloader, model, loss_fn, optimizer, device, verbose=False, callbacks=None, retain_graph=False, filter_classes=None, *args, **kwargs):
    num_batches = len(dataloader)
    model.train()
    optimizer.zero_grad()
    train_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        if filter_classes is not None:
            filtered_indices = torch.isin(torch.argmax(y, dim=1), torch.tensor(filter_classes)).nonzero().flatten()
            X, y = X[filtered_indices], y[filtered_indices]

        predictions = model(X)
        if len(y.shape) > 1:
            y = y[:predictions.shape[0], :predictions.shape[1]] 

        loss = loss_fn(predictions, y)
        train_loss += loss.item()
        loss.backward(retain_graph=retain_graph)
        if callbacks: 
            for callback in callbacks:
                callback(model=model, loss=loss, inputs=X, predictions=predictions, labels=y, *args, **kwargs)
                
        optimizer.step()
        optimizer.zero_grad()

    train_loss /= num_batches
    if verbose: print(f"Train Avg loss: {train_loss:>8f}")
    return train_loss

def test(dataloader, model, loss_fn, device, verbose=False, calculate_gradients=False, callbacks=None, retain_graph=False, filter_classes=None, *args, **kwargs):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        if filter_classes is not None:
            filtered_indices = torch.isin(torch.argmax(y, dim=1), torch.tensor(filter_classes)).nonzero().flatten()
            X, y = X[filtered_indices], y[filtered_indices]

        predictions = model(X)
        if len(y.shape) > 1:
            y = y[:predictions.shape[0], :predictions.shape[1]] 

        loss = loss_fn(predictions, y)
        test_loss += loss.item()
        if calculate_gradients: loss.backward(retain_graph=retain_graph)
        if callbacks: 
            for callback in callbacks:
                callback(model=model, loss=loss, inputs=X, predictions=predictions, labels=y, *args, **kwargs)

    test_loss /= num_batches
    if verbose: print(f"Test Avg loss: {test_loss:>8f}\n")    
    return test_loss

def extract_samples(dataloader, filter_classes=[], *args, **kwargs):
    inputs = []; labels = []
    for batch_inputs, batch_labels in dataloader:
        if filter_classes:
            filtered_indices = torch.isin(torch.argmax(batch_labels, dim=1), torch.tensor(filter_classes)).nonzero().flatten()
            batch_inputs, batch_labels = batch_inputs[filtered_indices], batch_labels[filtered_indices]
            batch_labels = batch_labels[:, :max(filter_classes) + 1]
            
        inputs.append(batch_inputs); labels.append(batch_labels)
    
    return torch.concatenate(inputs), torch.concatenate(labels)

def accuracy(predictions, labels): 
    "https://stackoverflow.com/questions/51503851/calculate-the-accuracy-every-epoch-in-pytorch/63271002#63271002"
    if len(predictions.shape) == 1 or predictions.size(1) == 1:
        # Binary classification
        return ((torch.sign(predictions) + 1.) * 0.5  == labels).sum() / predictions.size(0)
    
    else:
        # Multiclass classification
        return (torch.argmax(predictions, dim=1) == torch.argmax(labels, dim=1)).sum() / predictions.size(0)
    

class ExponentialLoss(object):
    def __call__(self, logits, labels): 
        return  (( -( (2.0 * labels.float() - 1.0) * logits ) ).exp()).mean()
    