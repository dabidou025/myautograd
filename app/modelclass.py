import inspect
import math
import time

from tensor import Tensor, to_tensor
from layers import Layer

class ModelClass:
    def parameters(self):
        for name, value in inspect.getmembers(self):
            if isinstance(value, Tensor):
                yield value
            if isinstance(value, Layer):
                #print(value.parameters())
                yield from value.parameters()
            elif isinstance(value, ModelClass):
                yield from value.parameters()

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()

    def fit (self, x, y, loss, optimizer, n_epochs=1, batch_size=None):

        x = to_tensor(x)
        y = to_tensor(y)

        if batch_size is None:
            batch_size = len(x)

        start = time.time()

        for epoch in range(1, n_epochs+1):
            loss_epoch = 0
            for i in range(0, x.shape[0], batch_size):
                self.zero_grad()

                pred = self.predict(x[i:(i+batch_size), ...])
                loss_tensor = loss(pred, y[i:(i+batch_size), ...])

                loss_epoch += loss_tensor.data

                loss_tensor.backward()

                optimizer.step(self)

            if epoch == 1 or epoch % math.floor(n_epochs / 10) == 0:
                print("Epoch ", epoch, "| Loss = {:.2f}".format(loss_epoch))

        end = time.time()
        print("Took {:.2f}".format(end - start), "seconds to fit")
