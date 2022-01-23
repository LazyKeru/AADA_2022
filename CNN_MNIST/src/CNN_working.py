import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Default values
_TAILLE_BATCH = 100
_EPOCH_NBR = 3
_LEARNING_RATE = 0.001

# Normalization is a pre-processing technique used to standardize data
#There are two main methods to normalize our data.
#One of the most straightforward method is to scale it to a range from 0 to 1:
# xn = (x - m)/(xmax-xmin)
#we are using tf's batch_normalization()
def normalisation(previous_layer):
    # create a norm layer after each convolution layer and first fully connected layer
    mean, var=tf.nn.moments(previous_layer, [0])
    scale=tf.Variable(tf.ones(shape=(np.shape(previous_layer)[-1])))
    beta=tf.Variable(tf.zeros(shape=(np.shape(previous_layer)[-1])))
    result=tf.nn.batch_normalization(previous_layer, mean, var, beta, scale, 0.001)
    return result


# Convolution Layer
def convolution(previous_layer, size_core, nbr_core):
    # previous_layer -- x w --> result_conv --> + b --> result
    # Creation of weights by initializing them according to a normal law
    w=tf.Variable(tf.random.truncated_normal(shape=(size_core, size_core, int(previous_layer.get_shape()[-1]), nbr_core)))
    # Creation of the bias by initializing them to 0
    b=np.zeros(nbr_core)
    # Definition of the convolution operation on the layer_prec and adding the bias
    result=tf.nn.conv2d(previous_layer, w, strides=[1, 1, 1, 1], padding='SAME')+b
    return result

# Fully Connected Layer to give prediction
def fc(previous_layer, nbr_neurone):
    w=tf.Variable(tf.random.truncated_normal(shape=(int(previous_layer.shape[-1]), nbr_neurone), dtype=tf.float32))
    b=tf.Variable(np.zeros(shape=(nbr_neurone)), dtype=tf.float32)
    result=tf.matmul(previous_layer, w)+b
    return result

def CNN(X_train,
        X_test,
        y_train,
        y_test,
        taille_batch = _TAILLE_BATCH,
        epoch_nbr = _EPOCH_NBR,
        learning_rate = _LEARNING_RATE
        ):

    tf.compat.v1.disable_eager_execution()

    # Entry (None, 28, 28, 1)
    ph_images=tf.compat.v1.placeholder(shape=(None, 28, 28, 1), dtype=tf.float32, name='images')
    # Exit (None, 10)
    ph_labels=tf.compat.v1.placeholder(shape=(None, 10), dtype=tf.float32)

    """
        DEBUT ARCHIECTURE DU RESEAU CONVOLUTIF
    """
    result=convolution(ph_images, 5, 16)
    result=normalisation(result)
    result=tf.nn.relu(result)
    result=convolution(result, 5, 16)
    result=normalisation(result)
    result=tf.nn.relu(result)
    result=tf.nn.max_pool(result, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    result=convolution(result, 5, 32)
    result=normalisation(result)
    result=tf.nn.relu(result)
    result=convolution(result, 5, 32)
    result=normalisation(result)
    result=tf.nn.relu(result)
    result=tf.nn.max_pool(result, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    #REplaces the deprecated flatten
    result=tf.reshape(result, [-1, result.shape[1]*result.shape[2]*result.shape[3]])

    result=fc(result, 512)
    result=normalisation(result)
    result=tf.nn.sigmoid(result)
    result=fc(result, 10)
    scso=tf.nn.softmax(result,name='sortie')
    """
        FIN ARCHIECTURE DU RESEAU CONVOLUTIF
    """

    loss=tf.nn.softmax_cross_entropy_with_logits(labels=ph_labels, logits=result)
    accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(scso, 1), tf.argmax(ph_labels, 1)), tf.float32))
    train=tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)
    #train=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

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
    pass
