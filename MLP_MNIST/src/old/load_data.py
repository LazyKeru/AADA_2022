import os
import gzip
import numpy as np

from tensorflow.python.platform import gfile

# MNIST constants
# CVDF mirror of http://yann.lecun.com/exdb/mnist/
_MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
_MNIST_TRAIN_DATA_FILENAME = "train-images-idx3-ubyte.gz"
_MNIST_TRAIN_LABELS_FILENAME = "train-labels-idx1-ubyte.gz"
_MNIST_TEST_DATA_FILENAME = "t10k-images-idx3-ubyte.gz"
_MNIST_TEST_LABELS_FILENAME = "t10k-labels-idx1-ubyte.gz"
_MNIST_IMAGE_SIZE = 28
MNIST_IMAGE_SHAPE = (_MNIST_IMAGE_SIZE, _MNIST_IMAGE_SIZE, 1)
MNIST_NUM_CLASSES = 10
_TRAIN_EXAMPLES = 60000
_TEST_EXAMPLES = 10000

#
path_mnist_train_images = (os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),'dataset/train-images-idx3-ubyte.gz'))
#
path_mnist_train_labels = (os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),'dataset/train-labels-idx1-ubyte.gz'))
#
path_mnist_test_images = (os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),'dataset/t10k-images-idx3-ubyte.gz'))
#
path_mnist_test_labels = (os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),'dataset/t10k-labels-idx1-ubyte.gz'))

def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_labels(f, one_hot=False, num_classes=10):
  """Extract the labels into a 1D uint8 numpy array [index].

  Args:
    f: A file object that can be passed into a gzip reader.
    one_hot: Does one hot encoding for the result.
    num_classes: Number of classes for the one hot encoding.

  Returns:
    labels: a 1D uint8 numpy array.

  Raises:
    ValueError: If the bystream doesn't start with 2049.
  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, f.name))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = np.frombuffer(buf, dtype=np.uint8)
    if one_hot:
      return dense_to_one_hot(labels, num_classes)
    return labels

def extract_images(path):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth].

  Args:
    f: A file object that can be passed into a gzip reader.

  Returns:
    data: A 4D uint8 numpy array [index, y, x, depth].

  Raises:
    ValueError: If the bytestream does not start with 2051.

  """
  with gfile.Open(path, 'rb') as f:
      print('Extracting', f.name)
      with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
          raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                           (magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        images = np.frombuffer(buf, dtype=np.uint8)
        images = images.reshape(num_images, rows, cols, 1)
        # reshape
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)
        return images

def load_data():
    mnist_train_images=extract_images(path_mnist_train_images)
    #mnist_train_images=np.fromfile(parse_mnist_file(path_mnist_train_images), dtype=np.uint8)[16:].reshape(-1, 784)/255
    # Dividing by 255 allows our network input to have values ​​between 0 and 1.
    #mnist_train_labels=np.eye(10)[np.fromfile(parse_mnist_file(path_mnist_train_labels), dtype=np.uint8)[8:]]
    mnist_test_images=extract_images(path_mnist_test_images)
    #mnist_test_labels=np.eye(10)[np.fromfile(parse_mnist_file(path_mnist_test_labels), dtype=np.uint8)[8:]]
    print(f"number of images in the train dataset: {mnist_train_images}")
    print(f"number of images in the test dataset: {mnist_test_images}")
    pass

load_data()
