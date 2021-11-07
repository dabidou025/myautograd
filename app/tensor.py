from collections import namedtuple
import numpy as np
from numpy.core.fromnumeric import shape
from numpy.core.records import array

Dependency = namedtuple('Dependency', ['tensor', 'grad_fn'])

def to_array (arrayable):
    return arrayable if isinstance(arrayable, np.ndarray) else np.array(arrayable)

def to_array_column (arrayable):
    ret = arrayable if isinstance(arrayable, np.ndarray) else np.array(arrayable)
    if ret.ndim > 1:
        raise RuntimeError('TensorColumn argument must be 1-dimensional')
    return np.reshape(ret, newshape=(ret.shape[0], 1))

def to_tensor (tensorable):
    return tensorable if isinstance(tensorable, Tensor) else Tensor(tensorable)

class Tensor:
    def __init__(self, data, requires_grad = False, depends_on = None):
        self.data = to_array(data)
        self.shape = self.data.shape
        
        self.depends_on = depends_on or []
        self.requires_grad = requires_grad

        self.grad = None

        if self.requires_grad:
            self.zero_grad()

    def zero_grad (self):
        self.grad = Tensor(np.zeros_like(self.data))

    def backward(self, grad = None):
        if not self.requires_grad:
            raise RuntimeError("Called backward on non-requires_grad tensor")

        if grad is None:
            if self.shape == ():
                grad = Tensor(1)
            else:
                raise RuntimeError("grad must be specified for non-0-tensor")

        self.grad.data = self.grad.data + grad.data

        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))

    def sum(self):
        return tensor_sum(self)
    
    def __add__ (self, other):
        return add(self, to_tensor(other))
    
    def __radd__ (self, other):
        return add(to_tensor(other), self)

    def __neg__ (self):
        return neg(self)

    def __sub__ (self, other):
        return sub(self, to_tensor(other))
    
    def __rsub__ (self, other):
        return sub(to_tensor(other), self)

    def __mul__ (self, other):
        return mul(self, to_tensor(other))
    
    def __rmul__ (self, other):
        return mul(to_tensor(other), self)

    def __matmul__ (self, other):
        return matmul(self, to_tensor(other))
        
    def __rmatmul__ (self, other):
        return matmul(to_tensor(other), self)

    def __iadd__ (self, other):
        return self + other

    def __isub__ (self, other):
        return self - other

    def __imul__ (self, other):
        return self * other

    def __getitem__(self, idxs):
        return slice(self, idxs)

class TensorColumn (Tensor):
    def __init__(self, data, requires_grad = False, depends_on = None):
        self.data = to_array_column(data)
        self.shape = self.data.shape
        
        self.depends_on = depends_on or []
        self.requires_grad = requires_grad

        self.grad = None

        if self.requires_grad:
            self.zero_grad()

def tensor_sum (t):
    data = t.data.sum()
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn (grad):
            return grad * np.ones_like(t.data)
        
        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)

def add (t1, t2):
    data = t1.data + t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on = []

    if t1.requires_grad:
        def grad_fn1 (grad):
            
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad
        
        depends_on.append(Dependency(t1, grad_fn1))
    
    if t2.requires_grad:
        def grad_fn2 (grad):

            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad
        
        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data, requires_grad, depends_on)

def neg (t):
    data = -t.data
    requires_grad = t.requires_grad
    
    if requires_grad:
        def grad_fn (grad):
            return -grad
        
        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)

def sub(t1, t2):
    return add(t1, neg(t2))

def mul (t1, t2):
    data = t1.data * t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on = []

    if t1.requires_grad:
        def grad_fn1 (grad):
            grad = grad * t2.data
            
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad
        
        depends_on.append(Dependency(t1, grad_fn1))
    
    if t2.requires_grad:
        def grad_fn2 (grad):
            grad = grad * t1.data

            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad
        
        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data, requires_grad, depends_on)

def matmul (t1, t2):
    data = t1.data @ t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on = []

    if t1.requires_grad:
        def grad_fn1 (grad):
            #print("t1 ", grad.shape, t2.data.T.shape)
            grad = grad @ t2.data.T
            return grad
        
        depends_on.append(Dependency(t1, grad_fn1))
    
    if t2.requires_grad:
        def grad_fn2 (grad):
            #print("t2 ", grad.shape, t1.data.T.shape)
            grad = t1.data.T @ grad
            return grad
        
        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data, requires_grad, depends_on)

def ReLu (t):
    data = np.maximum(t.data, 0)
    requires_grad = t.requires_grad
    depends_on = []

    if requires_grad:
        def grad_fn (grad):
            return grad * (data > 0).astype("float")
        
        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)

def Tanh (t):
    data = np.tanh(t.data)
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn(grad):
            return grad * (1 - data * data)

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)

def slice(t, idxs):
    data = t.data[idxs]
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn(grad):
            bigger_grad = np.zeros_like(data)
            bigger_grad[idxs] = grad
            return bigger_grad

        depends_on = Dependency(t, grad_fn)
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)

def squeeze (t, axis):
    data = np.squeeze(t.data, axis=axis)
    requires_grad = t.requires_grad
    depends_on = []

    if requires_grad:
        def grad_fn(grad):
            return np.expand_dims(grad, axis=axis)

        depends_on = [Dependency(t, grad_fn)]

    return Tensor(data, requires_grad, depends_on)