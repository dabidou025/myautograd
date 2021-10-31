import numpy as np

from tensor import *

if __name__ == "__main__":
    x = Tensor(np.ones(10), requires_grad=True)

    n_epochs = 100; lr = 1e-1
    for i in range(n_epochs):
        soq = tensor_sum(mul(x,x))
        soq.backward()

        delta = mul(Tensor(lr), x.grad)
        x = Tensor(x.data - delta.data, requires_grad=True)

        print(i, soq.data)