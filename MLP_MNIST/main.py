# main.py
import os
import src

path_mnist_train_images = (os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),'dataset/train-images-idx3-ubyte.gz'))
#
path_mnist_train_labels = (os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),'dataset/train-labels-idx1-ubyte.gz'))
#
path_mnist_test_images = (os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),'dataset/t10k-images-idx3-ubyte.gz'))
#
path_mnist_test_labels = (os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),'dataset/t10k-labels-idx1-ubyte.gz'))


def load_data():
    print(f"loading all dataset")
    mnist_train_images = src.load_images(path_mnist_train_images)/255
    mnist_train_labels = src.load_labels(path_mnist_train_labels)
    mnist_test_images = src.load_images(path_mnist_test_images)/255
    mnist_test_labels = src.load_labels(path_mnist_test_labels)
    print(f"loading all dataset")
    return mnist_train_images, mnist_test_images, mnist_train_labels, mnist_test_labels


#src.print_image(src.load_image(path_mnist_train_images, 16))

X_train, X_test, y_train, y_test = load_data()

## The shapes
print(X_train.shape) # (60000, 28, 28)
print(y_train.shape) # (60000, 10)
print(X_test.shape) # (10000, 28, 28)
print(y_test.shape) # (10000, 10)
## For X_train we want (60000, 784)
X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)
print(X_train.shape) # (60000, 784)
print(X_test.shape) # (10000, 784)

src.test(X_train, X_test, y_train, y_test)

# src.print_images(X_test[10:15])
