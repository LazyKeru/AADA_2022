# layers

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

def normalisation(previous_layer):
    """
    Normalization is a pre-processing technique used to standardize data. There are two main methods to normalize our data.
    One of the most straightforward method is to scale it to a range from 0 to 1: xn = (x - m)/(xmax-xmin)
    We will be using tf's batch_normalization() function
    :param tf previous_layer: previous layers
    :return: tf
    """
    # computes the mean and variance
    mean, var=tf.nn.moments(previous_layer, [0])
    # Set Scale and beta to respectively 1 and 0 so they don't influence the normalization
    scale=tf.Variable(tf.ones(shape=(np.shape(previous_layer)[-1])))
    beta=tf.Variable(tf.zeros(shape=(np.shape(previous_layer)[-1])))
    # factor to induce variance (noise) to the normalization
    result=tf.nn.batch_normalization(previous_layer, mean, var, beta, scale, 0.001)
    return result

def convolution(previous_layer, kernel=[2, 2], layers=16):
    """
    Convolutional layers are the layers where filters are applied to the original image, or to other feature maps in a deep CNN.
    The most important parameters are the number of kernels and the size of the kernels.
    We will be using tf's batch_normalization() function
    :param tf previous_layer: previous layers
    :return: tf
    """
    # Creation of weights by initializing them according to a normal law
    weights=tf.Variable(tf.random.truncated_normal(shape=(
                                                    kernel[0],
                                                    kernel[1],
                                                    int(previous_layer.get_shape()[-1]),
                                                    layers
                                                )))
    # Creation of the bias by initializing them to 0
    biases=np.zeros(layers)
    # Definition of the convolution operation on the layer_prec and adding the bias
    result=tf.nn.conv2d(previous_layer, weights, strides=[1, 1, 1, 1], padding='SAME')+biases
    return result

# Fully Connected Layer to give prediction
def fc(previous_layer, nbr_neurone):
    """
    Fully Connected layers in a neural networks are those layers where all the inputs from one layer are connected to every activation unit of the next layer.
    In most popular machine learning models, the last few layers are full connected layers which compiles the data extracted by previous layers to form the final output.
    :param tf previous_layer: previous layers
    :return: tf
    """
    weights=tf.Variable(tf.random.truncated_normal(shape=(int(previous_layer.shape[-1]), nbr_neurone), dtype=tf.float32))
    biases=tf.Variable(np.zeros(shape=(nbr_neurone)), dtype=tf.float32)
    result=tf.matmul(previous_layer, weights) + biases
    return result

def maxpool(previous_layer, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding='SAME'):
    """
    Max Pooling is a downsampling strategy in Convolutional Neural Networks.
    Let's say we have a 4x4 matrix representing our initial input. Let's say, as well, that we have a 2x2 filter that we'll run over our input. We'll have a stride of 2 (meaning the (dx, dy) for stepping over our input will be (2, 2)) and won't overlap regions. For each of the regions represented by the filter, we will take the max of that region and create a new, output matrix where each element is the max of a region in the original input.
    :param tf previous_layer: previous layers
    :return: tf
    """
    result = tf.nn.max_pool2d(
                                    input=previous_layer,
                                    strides=strides,
                                    ksize=ksize,
                                    padding=padding
                                )
    return result
