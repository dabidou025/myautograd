from modelclass import ModelClass

class SGD:
    def __init__(self, lr = 1e-2) -> None:
        self.lr = lr

    def step(self, model: ModelClass) -> None:
        for param in model.parameters():
            param -= self.lr * param.grad