import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def derivee_sigmoid(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)
def derivee_relu(x):
    return (x > 0).astype(float)

class simplemlp:
    def __init__(self, nb_neurones_entree, couches_cachees, nb_neurones_sortie, taux_apprentissage=0.01):
                self.taux_apprentissage = taux_apprentissage
                self.couches = [nb_neurones_entree] + couches_cachees + [nb_neurones_sortie]
                self.poids = []
                self.biais = []
                for i in range(len(self.couches) - 1):
                    self.poids.append(np.random.randn(self.couches[i], self.couches[i+1]) * 0.01)
                    self.biais.append(np.zeros((1, self.couches[i+1])))     
                    
    def propagation_avant(self, X):
        activations = [X]
        valeurs_Z = []
        for i in range(len(self.poids)):
            Z = np.dot(activations[-1], self.poids[i]) + self.biais[i]
            valeurs_Z.append(Z)
            if i < len(self.poids) - 1:
                A = relu(Z)  # Activation ReLU pour les couches cachées
            else:
                A = sigmoid(Z)  # Activation sigmoïde pour la couche de sortie
            activations.append(A)
            
            print("iteration ", i, ":")
            print(A)  
        return A
    
    
    def retropropagation(self, X, Y):
        activations, valeurs_Z = self.propagation_avant(X)
        dW = [None] * len(self.poids)
        dB = [None] * len(self.biais)
        dA = activations[-1] - Y  # Erreur de la sortie
        m = Y.shape[0]  # Nombre d'exemples 
        
        
               
        for i in reversed(range(len(self.poids))):
            if i == len(self.poids) - 1:
                dZ = dA * derivee_sigmoid(activations[i + 1])  # Dérivée pour la couche de sortie
            else:
                dZ = dA * derivee_relu(valeurs_Z[i])  # Dérivée pour les couches cachées
            print("iteration ", i, ":")
            print(dZ) 
            dW[i] = np.dot(activations[i].T, dZ) / m  # Gradient des poids
            dB[i] = np.sum(dZ, axis=0, keepdims=True) / m  # Gradient des biais
            dA = np.dot(dZ, self.poids[i].T)  # Propagation de l'erreur vers l'arrière      
        
        
                # Mise à jour des poids et biais
        for i in range(len(self.poids)):
            self.poids[i] -= self.taux_apprentissage * dW[i]
            self.biais[i] -= self.taux_apprentissage * dB[i]
            
            
    
                
                
 
    
mlp = simplemlp(nb_neurones_entree=4, couches_cachees=[10, 5], nb_neurones_sortie=1)
test = np.array([[0.2, 0.5, 0.8, 0.7]]) 
 
 
Z = mlp.propagation_avant(test)
print("Sortie du resau de neurone:", Z)        