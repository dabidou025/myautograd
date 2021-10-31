import numpy as np

from tensor import *

if __name__ == "__main__":
    x = Tensor(np.ones(3), requires_grad=True)

    n_epochs = 25; lr = 1e-1
    for i in range(n_epochs):
        x.zero_grad()
        
        soq = (x * x).sum()
        soq.backward()

        x -= lr * x.grad

        print(i, soq.data, x.data)