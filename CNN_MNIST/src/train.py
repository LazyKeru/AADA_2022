# train

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Default values
_TAILLE_BATCH = 100
_EPOCH_NBR = 3
_LEARNING_RATE = 0.001

def train(
        X_train,
        X_test,
        y_train,
        y_test,
        result,
        scso,
        ph_images,
        ph_labels,
        taille_batch = _TAILLE_BATCH,
        epoch_nbr = _EPOCH_NBR,
        learning_rate = _LEARNING_RATE
        ):

        loss=tf.nn.softmax_cross_entropy_with_logits(labels=ph_labels, logits=result)
        accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(scso, 1), tf.argmax(ph_labels, 1)), tf.float32))
        train=tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)

        """
            START TRAINING
        """
        with tf.compat.v1.Session() as s:

            # Initialisation des variables
            s.run(tf.compat.v1.global_variables_initializer())

            # To have a clear view of our session progress
            tab_train=[]
            tab_test=[]

            for id_entrainement in np.arange(epoch_nbr):
                print("ID entrainement", id_entrainement)
                tab_accuracy_train=[]
                tab_accuracy_test=[]
                for batch in np.arange(0, len(X_train), taille_batch):
                    # lancement de l'apprentissage en passant la commande "train"
                    s.run(train, feed_dict={
                        ph_images: X_train[batch:batch+taille_batch],
                        ph_labels: y_train[batch:batch+taille_batch]
                    })
                    precision=s.run(accuracy, feed_dict={
                        ph_images: X_train[batch:batch+taille_batch],
                        ph_labels: y_train[batch:batch+taille_batch]
                    })
                    tab_accuracy_train.append(precision)
                    print(f"precision: {precision}")
                    pass

                for batch in np.arange(0, len(X_test), taille_batch):
                    # lancement de la prédiction en passant la commande "accuracy"
                    precision=s.run(accuracy, feed_dict={
                        ph_images: X_test[batch:batch+taille_batch],
                        ph_labels: y_test[batch:batch+taille_batch]
                    })
                    tab_accuracy_test.append(precision)
                    print(f"precision: {precision}")
                    pass
                print("  train:", np.mean(tab_accuracy_train))
                tab_train.append(1-np.mean(tab_accuracy_train))
                print("  test :", np.mean(tab_accuracy_test))
                tab_test.append(1-np.mean(tab_accuracy_test))
                pass
            """
                END TRAINING
            """
        return s, tab_train, tab_test
