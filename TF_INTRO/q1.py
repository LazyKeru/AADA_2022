from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

iris = datasets.load_iris()

# Permet d'obtenir un vecteur vertical des data
Largeur = iris["data"][:, 3:]

# On créé une liste qui contient 1 lorsque la fleur est de type 2 et 0 sinon pour faire une classification
Type = (iris["target"] == 2).astype(np.int)
# print("Type : ", Type)

# On va programmer une résolution par descente de gradients 'à la main'
# Paramètre de la descente de gradient
learning_rate = 0.01
display_step = 1
n_epochs = 10000
with tf.compat.v1.Session() as sess:
    # Initialisation des tenseurs de constantes
    # numpy.ones Returns a new array of given shape and type, filled with ones.
    X = tf.constant(np.c_[np.ones((len(Largeur),1)), Largeur],dtype=tf.float32, name = "X") #X est un tensor qui contient les features 'Largeur' et une colonne de '1' pour le Theta0
    # print("X: ",np.c_[np.ones((len(Largeur),1)), Largeur])
    y = tf.constant(Type, shape = (len(Largeur),1), dtype = tf.float32, name="y") # y est un tensor qui représente deux classes possibles
    # print("Y: ",Type)


    # Modèle
    # theta est un tensor de 2 variables en colonne initialisées aléatoirement entre -1 et +1
    theta = tf.Variable(tf.random.uniform([2,1], -1.0, 1.0),  name = "theta")

    # la prédicton est faite avec la fonction logistique, pred est le tensor de toutes les prédictions
    pred = tf.sigmoid(tf.matmul(X,theta))

    # l'error est le tensor de toutes les erreurs de prédictions
    error = pred - y

    # calcule la MSE qui est en fait la valeur que minimise une descente de gradient sur une fonction logistique
    mse = tf.reduce_mean(tf.square(error), name="mse")
    nbExemples = len(Largeur)

    # Calcul du gradient de l'erreur
    gradients = (2/nbExemples) * tf.matmul(tf.transpose(X), error)

    # Definition de la fonction de correction de theta à partir du gradient
    training_op = tf.compat.v1.assign(theta, theta - learning_rate * gradients)
    init = tf.compat.v1.global_variables_initializer() # créer un noeud init dans le graphe qui correspond à l'initialisation

    # Execution du modèle

    sess.run(init)
    for epoch in range(n_epochs):
        # affichage tous les 100 pas de calcul
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        # Exécution d'un pas de recalcule de theta avec appels de tous les opérateurs et tensors nécessaires dans le graphe
        sess.run(training_op)
    best_theta = theta.eval()
    print("Best theta : ", best_theta)
