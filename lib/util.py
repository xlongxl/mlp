import numpy as np

def logistic(X):
    return 1.0 / (1.0 + np.exp(-X))

def logistic_deriv(X):
    Y = logistic(X)
    return Y * (1.0 - Y)

def square_err(Xe, X, scale=1.0):
    return scale * np.sum((Xe - X) ** 2)

def square_err_deriv(Xe, X, scale=1.0):
    return scale * 2 * (Xe - X)
