from .layers import convolution, normalisation, maxpool, fc
import tensorflow as tf

# model_builder

def default_CNN():
    tf.compat.v1.disable_eager_execution()
    ph_images=tf.compat.v1.placeholder(shape=(None, 28, 28, 1), dtype=tf.float32, name='images')
    ph_labels=tf.compat.v1.placeholder(shape=(None, 10), dtype=tf.float32)
    """
        DEBUT ARCHIECTURE DU RESEAU CONVOLUTIF
    """
    result= convolution(ph_images, (5,5), 16)
    result= normalisation(result)
    result=tf.nn.relu(result)
    result= convolution(ph_images, (5,5), 16)
    result= normalisation(result)
    result=tf.nn.relu(result)
    result= maxpool(result, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    result= convolution(ph_images, (5,5), 32)
    result= normalisation(result)
    result=tf.nn.relu(result)
    result= convolution(ph_images, (5,5), 32)
    result= normalisation(result)
    result=tf.nn.relu(result)
    result= maxpool(result, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    #REplaces the deprecated flatten
    result=tf.reshape(result, [-1, result.shape[1]*result.shape[2]*result.shape[3]])

    result= fc(result, 512)
    result= normalisation(result)
    result=tf.nn.sigmoid(result)
    result= fc(result, 10)
    scso=tf.nn.softmax(result,name='sortie')
    """
        FIN ARCHIECTURE DU RESEAU CONVOLUTIF
    """
    return result, scso, ph_images, ph_labels

def test1_CNN():
    tf.compat.v1.disable_eager_execution()
    ph_images=tf.compat.v1.placeholder(shape=(None, 28, 28, 1), dtype=tf.float32, name='images')
    ph_labels=tf.compat.v1.placeholder(shape=(None, 10), dtype=tf.float32)
    """
        DEBUT ARCHIECTURE DU RESEAU CONVOLUTIF
    """
    result= convolution(ph_images, (5,5), 16)
    result= normalisation(result)
    result=tf.nn.relu(result)
    result= maxpool(result, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    result= convolution(ph_images, (5,5), 32)
    result= normalisation(result)
    result=tf.nn.relu(result)
    result= maxpool(result, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    #REplaces the deprecated flatten
    result=tf.reshape(result, [-1, result.shape[1]*result.shape[2]*result.shape[3]])

    result= fc(result, 512)
    result= normalisation(result)
    result=tf.nn.sigmoid(result)
    result= fc(result, 10)
    scso=tf.nn.softmax(result,name='sortie')
    """
        FIN ARCHIECTURE DU RESEAU CONVOLUTIF
    """
    return result, scso, ph_images, ph_labels

def test2_CNN():
    tf.compat.v1.disable_eager_execution()
    ph_images=tf.compat.v1.placeholder(shape=(None, 28, 28, 1), dtype=tf.float32, name='images')
    ph_labels=tf.compat.v1.placeholder(shape=(None, 10), dtype=tf.float32)
    """
        DEBUT ARCHIECTURE DU RESEAU CONVOLUTIF
    """
    result= convolution(ph_images, (5,5), 16)
    result= normalisation(result)
    result=tf.nn.relu(result)
    result= convolution(ph_images, (5,5), 16)
    result= normalisation(result)
    result=tf.nn.relu(result)
    result= maxpool(result, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    #REplaces the deprecated flatten
    result=tf.reshape(result, [-1, result.shape[1]*result.shape[2]*result.shape[3]])

    result= fc(result, 512)
    result= normalisation(result)
    result=tf.nn.sigmoid(result)
    result= fc(result, 10)
    scso=tf.nn.softmax(result,name='sortie')
    """
        FIN ARCHIECTURE DU RESEAU CONVOLUTIF
    """
    return result, scso, ph_images, ph_labels
