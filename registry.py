from .src.losses.mse import MSELoss
from .src.optimizers.sgd import SGD

LOSS_REGISTRY = {
    "mse": MSELoss,
}

OPTIMIZER_REGISTRY = {
    "sgd": SGD,
}

def get_loss(identifier):
    if isinstance(identifier, str):
        return LOSS_REGISTRY[identifier.lower()]()
    return identifier

def get_optimizer(identifier):
    if isinstance(identifier, str):
        return OPTIMIZER_REGISTRY[identifier.lower()]()
    return identifier
