# layers


def normalisation(previous_layer):
    """
    Normalization is a pre-processing technique used to standardize data. There are two main methods to normalize our data.
    One of the most straightforward method is to scale it to a range from 0 to 1: xn = (x - m)/(xmax-xmin)
    We will be using tf's batch_normalization() function
    :param tf previous_layer: previous layers
    :return: tf
    """
    mean, var=tf.nn.moments(previous_layer, [0])
    scale=tf.Variable(tf.ones(shape=(np.shape(previous_layer)[-1])))
    beta=tf.Variable(tf.zeros(shape=(np.shape(previous_layer)[-1])))
    result=tf.nn.batch_normalization(previous_layer, mean, var, beta, scale, 0.001)
    return result

def convolution(previous_layer, size_core, nbr_core):
    """
    Normalization is a pre-processing technique used to standardize data. There are two main methods to normalize our data.
    One of the most straightforward method is to scale it to a range from 0 to 1: xn = (x - m)/(xmax-xmin)
    We will be using tf's batch_normalization() function
    :param tf previous_layer: previous layers
    :return: tf
    """
    # Creation of weights by initializing them according to a normal law
    w=tf.Variable(tf.random.truncated_normal(shape=(size_core, size_core, int(previous_layer.get_shape()[-1]), nbr_core)))
    # Creation of the bias by initializing them to 0
    b=np.zeros(nbr_core)
    # Definition of the convolution operation on the layer_prec and adding the bias
    result=tf.nn.conv2d(previous_layer, w, strides=[1, 1, 1, 1], padding='SAME')+b
    return result
