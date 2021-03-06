import numpy as np

from tensor import Tensor

class Parameter (Tensor):
    def __init__ (self, *shape):
        data = np.random.randn(*shape)
        super().__init__(data, requires_grad=True)
