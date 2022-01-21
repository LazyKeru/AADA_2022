import os
import gzip
import numpy as np
import matplotlib.pyplot as plt

# Default value
IMAGE_SIZE = 28

def load_image(path, num_images, image_size=IMAGE_SIZE):
    with gzip.open(path,'r') as f:
        f.read(num_images)
        buf = f.read(image_size * image_size * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, image_size, image_size, 1)
        print("data after load",data)
    return data

# np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784) // stack overflow - changing 784 to 783 // Internet
# np.fromfile("dataset/mnist/train-images.idx3-ubyte", dtype=np.uint8)[16:].reshape(-1, 784)/255 // Example not working
def load_images(path):
    with gzip.open(path, 'r') as f:
        print(f"Starting loading the images from {path}")
        # big' means big endian which defines the byte order (256)
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of images
        image_count = int.from_bytes(f.read(4), 'big')
        print(f"{path} has {image_count} images")
        # third 4 bytes is the row count
        row_count = int.from_bytes(f.read(4), 'big')
        print(f"The dataset has {row_count} row")
        # fourth 4 bytes is the column count
        column_count = int.from_bytes(f.read(4), 'big')
        print(f"The dataset has {column_count} column")
        # rest is the image pixel data, each pixel is stored as an unsigned byte
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8).astype(np.float32)\
            .reshape((image_count, row_count, column_count))
        print(images)
        print(f"Done loading the images from {path}")
        return images

def load_labels(path, one_hot = True):
    with gzip.open(path, 'r') as f:
        print(f"Starting loading the labels from {path}")
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of labels
        label_count = int.from_bytes(f.read(4), 'big')
        print(f"{path} has {label_count} labels")
        # rest is the label data, each label is stored as unsigned byte
        # label values are 0 to 9
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        # dense_to_one_hot labels_dense=labels num_classes=num_classes
        if one_hot == True:
            num_labels = labels.shape[0] # number of rows for our one_hot structure (number of items)
            index_offset = np.arange(num_labels) * 10
            labels_one_hot = np.zeros((num_labels, 10)) # creates the one_hot structure
            labels_one_hot.flat[index_offset + labels.ravel()] = 1 # fills the one_hot_structure
            labels = labels_one_hot
            pass
        print(labels)
        print(f"Done loading the labels from {path}")
        return labels
