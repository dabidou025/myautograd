from typing import NamedTuple, Callable, List, Union
import numpy as np

class Dependency(NamedTuple):
    tensor: "Tensor"
    grad_fn: Callable[[np.ndarray], np.ndarray]

Arrayable = Union[float, list, np.ndarray]

def to_array (arrayable: Arrayable) -> np.ndarray:
    return arrayable if isinstance(arrayable, np.ndarray) else np.array(arrayable)

Tensorable = Union["Tensor", float, np.ndarray]

def to_tensor (tensorable: Tensorable) -> np.ndarray:
    return tensorable if isinstance(tensorable, Tensor) else Tensor(tensorable)

class Tensor:
    def __init__(self,
                 data: Arrayable,
                 requires_grad: bool = False,
                 depends_on: List[Dependency] = None) -> None:
        self.data = to_array(data); self.shape = self.data.shape
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []
        self.grad: Tensor = None

        if self.requires_grad:
            self.zero_grad()

    def zero_grad (self) -> None:
        self.grad = Tensor(np.zeros_like(self.data))

    def backward(self, grad: "Tensor" = None) -> None:
        if not self.requires_grad:
            RuntimeError("Called backward on non-requires_grad tensor")

        if grad is None:
            if self.shape == ():
                grad = Tensor(1)
            else:
                RuntimeError("grad must be specified for non-0-tensor")

        self.grad.data = self.grad.data + grad.data

        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))

    def sum(self) -> "Tensor":
        return tensor_sum(self)
    
    def __add__ (self, other) -> "Tensor":
        return add(self, to_tensor(other))
    
    def __radd__ (self, other) -> "Tensor":
        return add(to_tensor(other), self)

    def __neg__ (self) -> "Tensor":
        return neg(self)

    def __sub__ (self, other) -> "Tensor":
        return sub(self, to_tensor(other))
    
    def __rsub__ (self, other) -> "Tensor":
        return sub(to_tensor(other), self)

    def __mul__ (self, other) -> "Tensor":
        return mul(self, to_tensor(other))
    
    def __rmul__ (self, other) -> "Tensor":
        return mul(to_tensor(other), self)

    def __matmul__ (self, other) -> "Tensor":
        return matmul(self, to_tensor(other))
        
    def __rmatmul__ (self, other) -> "Tensor":
        return matmul(to_tensor(other), self)

    def __iadd__ (self, other) -> "Tensor":
        return self + other

    def __isub__ (self, other) -> "Tensor":
        return self - other

    def __imul__ (self, other) -> "Tensor":
        return self * other

    def __getitem__(self, idxs) -> 'Tensor':
        return slice(self, idxs)

def tensor_sum (t: Tensor) -> Tensor:
    data = t.data.sum()
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn (grad: np.ndarray) -> np.ndarray:
            return grad * np.ones_like(t.data)
        
        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)

def add (t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data + t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on = []

    if t1.requires_grad:
        def grad_fn1 (grad: np.ndarray) -> np.ndarray:
            
            ndims_added = grad.ndim - t1.data.ndim;
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad
        
        depends_on.append(Dependency(t1, grad_fn1))
    
    if t2.requires_grad:
        def grad_fn2 (grad: np.ndarray) -> np.ndarray:

            ndims_added = grad.ndim - t2.data.ndim;
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad
        
        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data, requires_grad, depends_on)

def neg (t: Tensor) -> Tensor:
    data = -t.data
    requires_grad = t.requires_grad
    
    if requires_grad:
        def grad_fn (grad: np.ndarray) -> np.ndarray:
            return -grad
        
        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)

def sub(t1: Tensor, t2:Tensor) -> Tensor:
    return add(t1, neg(t2))

def mul (t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data * t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on = []

    if t1.requires_grad:
        def grad_fn1 (grad: np.ndarray) -> np.ndarray:
            grad = grad * t2.data
            
            ndims_added = grad.ndim - t1.data.ndim;
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad
        
        depends_on.append(Dependency(t1, grad_fn1))
    
    if t2.requires_grad:
        def grad_fn2 (grad: np.ndarray) -> np.ndarray:
            grad = grad * t1.data

            ndims_added = grad.ndim - t2.data.ndim;
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad
        
        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data, requires_grad, depends_on)

def matmul (t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data @ t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on = []

    if t1.requires_grad:
        def grad_fn1 (grad: np.ndarray) -> np.ndarray:
            #print("t1 ", grad.shape, t2.data.T.shape)
            grad = grad @ t2.data.T
            return grad
        
        depends_on.append(Dependency(t1, grad_fn1))
    
    if t2.requires_grad:
        def grad_fn2 (grad: np.ndarray) -> np.ndarray:
            #print("t2 ", grad.shape, t1.data.T.shape)
            grad = t1.data.T @ grad
            return grad
        
        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data, requires_grad, depends_on)

def ReLu (t: Tensor) -> Tensor:
    data = np.maximum(t.data, 0)
    requires_grad = t.requires_grad
    depends_on = []

    if requires_grad:
        def grad_fn (grad: np.ndarray) -> np.ndarray:
            return grad * (data > 0).astype("float")
        
        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)

def Tanh (t: Tensor) -> Tensor:
    data = np.tanh(t.data)
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (1 - data * data)

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)

def slice(t: Tensor, idxs) -> Tensor:
    data = t.data[idxs]
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            bigger_grad = np.zeros_like(data)
            bigger_grad[idxs] = grad
            return bigger_grad

        depends_on = Dependency(t, grad_fn)
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)