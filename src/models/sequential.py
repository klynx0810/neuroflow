from .model import Model
from ..layers.base import Layer
from typing import List, Optional

class Sequential(Model):
    def __init__(self, layers: Optional[List[Layer]] = None, name=None):
        super().__init__(name=name)
        if layers:
            for layer in layers:
                self.add(layer)
