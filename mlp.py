import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def relu(x):
    return np.maximum(0, x)

class simplemlp:
    def __init__(self, NmbrNeuronesEntrer, NmbrCouchesCachees, NmbrNeuronesSortie):
        self.couches = [NmbrNeuronesEntrer] + NmbrCouchesCachees + [NmbrNeuronesSortie]
        self.W = []
        self.B = []
        for i in range(len(self.couches) - 1):
            self.W.append(np.random.randn(self.couches[i], self.couches[i+1]) * 0.01)
            self.B.append(np.zeros((1, self.couches[i+1])))
