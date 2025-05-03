
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
                A = relu(Z)
            else:
                A = sigmoid(Z)
            activations.append(A)
            
             
        return activations, valeurs_Z
    
    
    def retropropagation(self, X, Y):
        activations, valeurs_Z = self.propagation_avant(X)
        dW = [0] * len(self.poids)
        dB = [0] * len(self.biais)
        dA = activations[-1] - Y  # Erreur de la sortie
        m = Y.shape[0]  # Nombre d'exemples 
        
        
           #  calculer les gradients des Wet de B pour chaque couche   
        for i in reversed(range(len(self.poids))):
           
            if i == len(self.poids) - 1:
                dZ = dA * derivee_sigmoid(activations[i+1])#calculer la deriveer pour  la cachee de sortie
            else:
                dZ = dA * derivee_relu(valeurs_Z[i])  #calculer la deriveer pour  les couvhes cachees
        
            dW[i] = np.dot(activations[i].T, dZ) / m  # Gradient des valeur de W
            dB[i] = np.sum(dZ, axis=0, keepdims=True) / m  # Gradient  de la valeur de B
            dA = np.dot(dZ, self.poids[i].T)  # Propagation de l'erreur vers l'arrière      
        
        
    # mise a jour des valeur de W et B 
        for i in range(len(self.poids)):
            self.poids[i] -= self.taux_apprentissage * dW[i]
            self.biais[i] -= self.taux_apprentissage * dB[i]
            
# l'entrainement de mlp
    def entrainer(self, X, Y, epochs=10000):
        for epoch in range(epochs):
            self.retropropagation(X, Y)
            if epoch % 1000 == 0:
                activations, _ = self.propagation_avant(X)
                erreur = np.mean((Y - activations[-1])**2)
                print(f"Epoch {epoch}, Erreur: {erreur:.4f}")
        niveau_apprentissage = (1 - erreur) * 100
        print(f"Niveau de apprentissage atteint : {niveau_apprentissage:.2f}%")        
        return erreur        


data = np.loadtxt("data.txt")
X = data[:, :-1]
Y = data[:, -1:]
X_train = X[:16000]
Y_train = Y[:16000]






mlp = simplemlp(nb_neurones_entree=3, couches_cachees=[8], nb_neurones_sortie=1, taux_apprentissage=10)
Resultat = mlp.entrainer(X_train,Y_train, epochs=10000)
print("errore de entrenments :", Resultat)

# Sauvegarde manuelle des poids et biais
dict_save = {}
for i, w in enumerate(mlp.poids):
    dict_save[f"poids_{i}"] = w
for i, b in enumerate(mlp.biais):
    dict_save[f"biais_{i}"] = b

np.savez("rachid.npz", **dict_save)














"""
activations, _ = mlp.propagation_avant(X_train)
#print("Sortie apres entrainement:\n", activations[-1][:10])
   



predictions = activations[-1]
pred_classes = (predictions > 0.5).astype(int)  


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


mask_1 = (pred_classes == 1).flatten()
ax.scatter(X_train[mask_1, 0], X_train[mask_1, 1], X_train[mask_1, 2], c='green', label='Classe 1')


mask_0 = (pred_classes == 0).flatten()
ax.scatter(X_train[mask_0, 0], X_train[mask_0, 1], X_train[mask_0, 2], c='red', label='Classe 0')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Résultats de classification en 3D')
ax.legend()
plt.show()
"""