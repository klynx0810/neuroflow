from .src.losses.mse import MSELoss
from .src.optimizers.sgd import SGD
from .src.layers.activations.relu import ReLU
from .src.layers.activations.sigmoid import Sigmoid
from .src.layers.activations.tanh import Tanh

LOSS_REGISTRY = {
    "mse": MSELoss,
}

OPTIMIZER_REGISTRY = {
    "sgd": SGD,
}

ACTIVATION_REGISTRY = {
    "relu": ReLU,
    "sigmoid": Sigmoid,
    "tanh": Tanh 
}

def get_loss(identifier):
    if isinstance(identifier, str):
        return LOSS_REGISTRY[identifier.lower()]()
    return identifier

def get_optimizer(identifier):
    if isinstance(identifier, str):
        return OPTIMIZER_REGISTRY[identifier.lower()]()
    return identifier

def get_activation(identifier):
    if isinstance(identifier, str):
        act_cls = ACTIVATION_REGISTRY.get(identifier.lower())
        if act_cls is None:
            raise ValueError(f"Unknown activation '{identifier}'")
        return act_cls()
    return identifier