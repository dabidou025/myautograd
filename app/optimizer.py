from modelclass import ModelClass

class SGD:
    def __init__ (self, lr=1e-2, beta=0):
        self.lr = lr
        self.beta = beta

    def step (self, model):
        for param in model.parameters():
            param.data -= self.lr * param.grad.data
            param.data -= self.lr * (self.beta * param.data + (1 - self.beta) * param.grad.data)
