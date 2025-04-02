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

    def Avantpropa(self, X):
        A = X
        for i in range(len(self.W)):
            Z = np.dot(A, self.W[i]) + self.B[i]
            if i < len(self.W) - 1:
                A = relu(Z) 
            else:
                A = sigmoid(Z) 
            print("iteration ", i, ":")
            print(A)
        return A
    
mlp = simplemlp(NmbrNeuronesEntrer=4, NmbrCouchesCachees=[10, 5], NmbrNeuronesSortie=1)
test = np.array([[0.2, 0.5, 0.8, 0.7]]) 
 
Z = mlp.Avantpropa(test)
print("Sortie du resau de neurone:", Z)
