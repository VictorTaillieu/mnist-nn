import numpy as np


def identity(x):
    return x


def identity_backward(x):
    return 1


def relu(x):
    return np.maximum(0, x)


def relu_backward(x):
    return (x > 0).astype(int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_backward(x):
    return sigmoid(x) * (1 - sigmoid(x))
