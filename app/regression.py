import time
import numpy as np
import pandas as pd
import plotly.express as px

from tensor import *
from parameter import Parameter
from modelclass import ModelClass
from optimizer import SGD

if __name__ == "__main__":

    n_sample = 1000
    #x = Tensor(2*np.random.randn(n_sample, 2))
    #y = np.sin(x.data[:,0]) + np.sin(x.data[:,1])
    #y = Tensor(np.vstack([y,y]).T)

    x = Tensor(2*np.random.randn(n_sample, 2))
    y = TensorColumn(np.sin(x.data[:,0]) + np.sin(x.data[:,1]))
    print('y', y.shape)

    class Model(ModelClass):
        def __init__(self, num_hidden = 20):
            self.w1 = Parameter(2, num_hidden)
            self.b1 = Parameter(num_hidden)

            self.w2 = Parameter(num_hidden, 1)
            self.b2 = Parameter(1)

        def predict(self, inputs):

            pred = inputs @ self.w1 + self.b1
            #print('pred', pred.shape)
            pred = Tanh(pred)
            #print('pred', pred.shape)
            pred = pred @ self.w2 + self.b2
            #print('pred', pred.shape)
            return pred

    model = Model()
    optimizer = SGD(lr=1e-5)

    n_epochs = 1000; lr = 1e-4

    start = time.time()

    batch_size = 100
    for epoch in range(n_epochs):
        loss_epoch = 0
        for i in range(0, n_sample, batch_size):
            model.zero_grad()

            pred = model.predict(x[i:(i+batch_size),:])
            err = y[i:(i+batch_size)] - pred
            loss = (err * err).sum()

            loss_epoch += loss.data

            loss.backward()

            model.w1.data -= lr * model.w1.grad.data
            model.b1.data -= lr * model.b1.grad.data

            model.w2.data -= lr * model.w2.grad.data
            model.b2.data -= lr * model.b2.grad.data
            #optimizer.step(model)

        if epoch % 100 == 0:
            print(epoch, loss_epoch)

    end = time.time()
    print(end - start)

    pred = model.predict(x)

    data_y = pd.DataFrame({"x0":x.data[:,0], "x1":x.data[:,1], "yval":y.data[:,0], "color":"greens"})
    data_pred = pd.DataFrame({"x0":x.data[:,0], "x1":x.data[:,1], "yval":pred.data[:,0], "color":"reds"})
    data = pd.concat([data_y, data_pred])

    fig = px.scatter_3d(data, x=data['x0'], y=data['x1'], z=data['yval'], color="color", size=np.ones(2*n_sample))
    fig.show()
