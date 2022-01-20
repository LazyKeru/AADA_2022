import tensorflow as tf
import numpy as np


def test(mnist_train_images, mnist_test_labels, mnist_train_labels, mnist_test_images):
    tf.compat.v1.disable_eager_execution()
    # Image entry
    ph_images=tf.compat.v1.placeholder(shape=(None, 784), dtype=tf.float32)

    # predicted labels results
    ph_labels=tf.compat.v1.placeholder(shape=(None, 10), dtype=tf.float32)

    # parameters

    nbr_ni=100
    learning_rate=0.0001
    taille_batch=100
    nbr_entrainement=200

    #  les différents poids du réseau wci, les différents biais bci
    wci=tf.Variable(tf.random.truncated_normal(shape=(784, nbr_ni)), dtype=tf.float32)
    bci=tf.Variable(np.zeros(shape=(nbr_ni)), dtype=tf.float32)

    # sci est d'abord le résultat de la somme pondrée des entrées. Ce résultat passe ensuite dans la fonction d'activation (ici sigmoid).
    sci=tf.matmul(ph_images, wci)+bci
    sci=tf.nn.sigmoid(sci)

    # pareil sortie
    wcs=tf.Variable(tf.random.truncated_normal(shape=(nbr_ni, 10)), dtype=tf.float32)
    bcs=tf.Variable(np.zeros(shape=(10)), dtype=tf.float32)
    scs=tf.matmul(sci, wcs)+bcs
    scso=tf.nn.softmax(scs)

    # nous devons définir une fonction de perte à optimiser (loss) et la méthode d'optimisation à utiliser (GradientDescentOptimizer). La fonction de loss est définie comme la crosse entropie. D'autres méthodes d'optimisation sont disponibles et vous pourrez les tester.
    loss=tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=ph_labels, logits=scs)
    train=tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(scso, 1), tf.argmax(ph_labels, 1)), dtype=tf.float32))

    with tf.compat.v1.Session() as s:

        # Initialisation des variables
        s.run(tf.global_variables_initializer())

        tab_acc_train=[]
        tab_acc_test=[]

        for id_entrainement in range(nbr_entrainement):
            print("ID entrainement", id_entrainement)
            for batch in range(0, len(mnist_train_images), taille_batch):
                # lancement de l'apprentissage en passant la commande "train". feed_dict est l'option désignant ce qui est
                # placé dans les placeholders
                s.run(train, feed_dict={
                    ph_images: mnist_train_images[batch:batch+taille_batch],
                    ph_labels: mnist_train_labels[batch:batch+taille_batch]
                })

            # Prédiction du modèle sur les batchs du dataset de training
            tab_acc=[]
            for batch in range(0, len(mnist_train_images), taille_batch):
                # lancement de la prédiction en passant la commande "accuracy". feed_dict est l'option désignant ce qui est
                # placé dans les placeholders
                acc=s.run(accuracy, feed_dict={
                    ph_images: mnist_train_images[batch:batch+taille_batch],
                    ph_labels: mnist_train_labels[batch:batch+taille_batch]
                })
                # création le tableau des accuracies
                tab_acc.append(acc)

            # calcul de la moyenne des accuracies
            print("accuracy train:", np.mean(tab_acc))
            tab_acc_train.append(1-np.mean(tab_acc))

            # Prédiction du modèle sur les batchs du dataset de test
            tab_acc=[]
            for batch in range(0, len(mnist_test_images), taille_batch):
                acc=s.run(accuracy, feed_dict={
                    ph_images: mnist_test_images[batch:batch+taille_batch],
                    ph_labels: mnist_test_labels[batch:batch+taille_batch]
                })
                tab_acc.append(acc)
            print("accuracy test :", np.mean(tab_acc))
            tab_acc_test.append(1-np.mean(tab_acc))
            resulat=s.run(scso, feed_dict={ph_images: mnist_test_images[0:taille_batch]})
