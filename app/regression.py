import time
import numpy as np
import pandas as pd
import plotly.express as px

from tensor import *
from parameter import Parameter
from modelclass import ModelClass
from optimizer import *
from layers import Dense
from losses import *

if __name__ == "__main__":

    n_sample = 1000

    x = Tensor(2*np.random.randn(n_sample, 2))
    y = TensorColumn(np.sin(x.data[:,0]) + np.sin(x.data[:,1]))

    class Model(ModelClass):
        def __init__(self, num_hidden = 100):
            self.layer1 = Dense(2, num_hidden)
            self.layer2 = Dense(num_hidden, 20)
            self.layer3 = Dense(20, 1)

        def predict(self, inputs):
            pred = self.layer1(inputs)
            pred = Tanh(pred)
            pred = self.layer2(pred)
            pred = Tanh(pred)
            pred = self.layer3(pred)
            return pred

    model = Model()
    sgd_optimizer = SGD(lr=1e-4, beta=0.5)

    model.fit(x, y, n_epochs=1000, batch_size=100, loss=MSE, optimizer=sgd_optimizer)

    pred = model.predict(x)

    if True:
        data_y = pd.DataFrame({"x0":x.data[:,0], "x1":x.data[:,1], "yval":y.data[:,0], "color":"greens"})
        data_pred = pd.DataFrame({"x0":x.data[:,0], "x1":x.data[:,1], "yval":pred.data[:,0], "color":"reds"})
        data = pd.concat([data_y, data_pred])

        fig = px.scatter_3d(data, x=data['x0'], y=data['x1'], z=data['yval'], color="color", size=np.ones(2*n_sample))
        fig.show()
