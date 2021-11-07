

def MSE (pred, y):
    err = y - pred
    return (err * err).sum() / float(y.shape[0])
