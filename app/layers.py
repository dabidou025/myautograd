from tensor import Tensor
from parameter import Parameter

from abc import ABC, abstractmethod
import inspect

class Layer (ABC):
    def __init__ (self, in_shape, out_shape):
        self.in_shape = in_shape
        self.out_shape = out_shape

    @abstractmethod
    def parameters():
        pass

class Dense (Layer):
    def __init__ (self, in_shape, out_shape):
        super().__init__(in_shape, out_shape)

        self.W = Parameter(in_shape, out_shape)
        self.bias = Parameter(out_shape)

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward (self, inputs):
        return inputs @ self.W + self.bias

    def parameters(self):
        for name, value in inspect.getmembers(self):
            if isinstance(value, Parameter):
                yield value
