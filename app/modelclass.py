import inspect

from tensor import Tensor   

class ModelClass:
    def parameters(self):
        for name, value in inspect.getmembers(self):
            if isinstance(value, Tensor):
                yield value
            elif isinstance(value, ModelClass):
                yield from value.parameters()

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()
