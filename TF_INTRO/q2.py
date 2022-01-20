from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

iris = datasets.load_iris()

# Décrit le dataset
print(iris.DESCR)

allDataset = iris["data"][:,:]

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
    # numpy.ones Returns a new array of given shape and type
    # numpy.c_ Translates slice objects to concatenation along the second axis.
    # printing arguments for X
    x_value = np.c_[np.ones((len(allDataset),1)), allDataset]
    x_dtype = tf.float32
    x_name = "X"
    print(f"""
        begin arg {x_name}
        name: {x_name}
        value: {x_value}
        dtype: {x_dtype}
        end arg {x_name}
    """)
    X = tf.constant(x_value, dtype= x_dtype, name = x_name) #X est un tensor qui contient les features 'sepal width', 'sepal length', 'petal  length', 'petal  width',  et une colonne de '1' pour le Theta0
    # printing arguments for y
    y_value = Type
    y_shape = (len(allDataset),1)
    y_dtype = tf.float32
    y_name = "y"
    print(f"""
        begin arg {y_name}
        name: {y_name}
        shape: {y_shape}
        value: {y_value}
        dtype: {y_dtype}
        end arg {y_name}
    """)
    y = tf.constant(y_value, shape = y_shape, dtype = y_dtype, name=y_name) # y est un tensor qui représente deux classes possibles


    # Modèle
    # theta est un tensor de 2 variables en colonne initialisées aléatoirement entre -1 et +1
    theta = tf.Variable(tf.random.uniform([5,1], -1.0, 1.0),  name = "theta")

    # la prédicton est faite avec la fonction logistique, pred est le tensor de toutes les prédictions
    # sigmoid(): Computes sigmoid of x element-wise
    # matmul(): Multiplies matrix a by matrix b
    pred = tf.sigmoid(tf.matmul(X,theta))

    # l'error est le tensor de toutes les erreurs de prédictions
    error = pred - y # prediction - la vrai valeur. no error: 0, yes error: 1 ou -1

    # calcule la MSE qui est en fait la valeur que minimise une descente de gradient sur une fonction logistique
    mse = tf.reduce_mean(tf.square(error), name="mse")
    nbExemples = len(allDataset)

    # Calcul du gradient de l'erreur
    gradients = (2/nbExemples) * tf.matmul(tf.transpose(X), error)

    # Definition de la fonction de correction de theta à partir du gradient
    training_op = tf.compat.v1.assign(theta, theta - learning_rate * gradients)
    init = tf.compat.v1.global_variables_initializer() # créer un noeud init dans le graphe qui correspond à l'initialisation

    # Execution du modèle
    loss = []
    sess.run(init)
    for epoch in range(n_epochs):
        # affichage tous les 100 pas de calcul
        if epoch % 100 == 0:
            loss.append(mse.eval())
        # Exécution d'un pas de recalcule de theta avec appels de tous les opérateurs et tensors nécessaires dans le graphe
        sess.run(training_op)
    best_theta = theta.eval()
    print("Best theta : ", best_theta)

    # Montre l'evolution de la perte pour les epochs
    print("loss : ", loss)
    plt.plot(loss)
    plt.show()
    # On va voir la prediction de notre modéle avec le meilleur theta
    # la prédicton est faite avec la fonction logistique, pred est le tensor de toutes les prédictions futures avec theta qui prend la valeur de "best_theta"

    # Changes the pred to best_theta
    pred = tf.sigmoid(tf.matmul(X,best_theta))

    yest = pred.eval()
    yhat = (pred.eval() > 0.5).astype(np.float)

    print("index, prediction")
    for (i, prediction) in enumerate(yhat[:,0] - Type, start=1):
        print(i, prediction)

    print(f"Faux classe 1 : {np.where(yhat[:,0] - Type > 0)}")
    print(f"Faux classe 0 : {np.where(yhat[:,0] - Type < 0)}")
    print(f"Vrai classe 1 : {np.where((yhat[:,0] - Type == 0) & (yhat[:,0] == 1))}")
    print(f"Vrai classe 0 : {np.where((yhat[:,0] - Type == 0) & (yhat[:,0] == 0))}")
