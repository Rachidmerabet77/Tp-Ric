import numpy as np

def relu(x):
    return np.maximum(0, x)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

data = np.load("rachid.npz")
nombre_couches = len([k for k in data.files if k.startswith("poids_")])
poids = [data[f"poids_{i}"] for i in range(nombre_couches)]
biais = [data[f"biais_{i}"] for i in range(nombre_couches)]


# Propagation avant
def propagation_avant(X, poids, biais):
    activation = X
    for i in range(len(poids)):
        Z = np.dot(activation, poids[i]) + biais[i]
        if i < len(poids) - 1:
            activation = relu(Z)
        else:
            activation = sigmoid(Z)
    return activation



X_new = np.loadtxt("me.txt")
Y_pred = propagation_avant(X_new, poids, biais)
print("le resultat est :" , Y_pred)
classes = (Y_pred > 0.5).astype(int)




# Sauvegarder les resultats dans un fichier texte
with open("resultats.txt", "w") as f:
    for x, y in zip(X_new, classes):
        ligne = ' '.join(map(str, x)) + f' {y[0]}\n'
        f.write(ligne)
print("Les resultats ont ete sauvegardes dans 'resultats.txt'")