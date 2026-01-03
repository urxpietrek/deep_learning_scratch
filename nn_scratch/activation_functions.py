import numpy as np

def linear(z):
    return z

def sigmoid(z):
    return 1/(1+np.exp(-z))

def dsigmoid(z):
    return sigmoid(z) * (1-sigmoid(z))
    
def tanh(z):
    return np.tanh(z)

def dtanh(z):
    return 1 - tanh(z)**2

def relu(z):
    return np.maximum(0, z)

def drelu(z):
    return np.where(z > 0, 1, 0)

def softmax(z):
    g = np.exp(z)
    return g / np.sum(g, axis=1).reshape(-1, 1)

def dsoftmax(z):
    return 