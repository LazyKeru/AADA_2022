import numpy as np
import matplotlib.pyplot as plt

def print_image(data):
    image = np.asarray(data[2]).squeeze()
    plt.imshow(image)
    plt.show()
    pass

def print_images(images):
    # Will be very long, as it prints every elements
    for index, image in enumerate(images):
        plt.imshow(image.reshape(28,28),cmap='gray')
        plt.show()
