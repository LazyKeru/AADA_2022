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
# Loading mnist image and normalizing the values (pixels grayscaled 0 - 255)

## The shapes
print(f"X_train shape: {X_train.shape}") # (60000, 28, 28)
print(f"y_train shape: {y_train.shape}") # (60000, 10)
print(f"X_test shape: {X_test.shape}") # (10000, 28, 28)
print(f"y_test shape: {y_test.shape}") # (10000, 10)
print("They are not the correct shape")
#X_train = X_train.reshape(X_train.shape[0], 784)
#X_test = X_test.reshape(X_test.shape[0], 784)
#X_train = X_train.reshape(-1, 28, 28, 1)
#X_test = X_test.reshape(-1, 28, 28, 1)

print(X_train) # (60000, 784)
print(X_test) # (10000, 784)

src.CNN(X_train, X_test, y_train, y_test)

# src.print_images(X_test[10:15])
