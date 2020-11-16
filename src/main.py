import numpy as np
import tensorflow as tf
from tensorflow import keras
import struct
from array import array
import sys
import random
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from kerasimport layers, optimizers, losses, metrics

def main(argv):
    if(len(sys.argv) != 3):
        print("Sorry, the input must be in this form: autoencoder.py –d <dataset>")
        exit(1)
    else:
        if(str(sys.argv[1]) != '-d'):
            print("Sorry, the input must be in this form: autoencoder.py –d <dataset>")
            exit(1)
        else:
            train_file = str(sys.argv[2])

    images = Load_Mnist_Images(train_file)
    # print(images[0][0])
    images_2_show = []
    titles_2_show = []
    for i in range(0, 20):
        r = random.randint(1, 60000)
        images_2_show.append(images[r])
        titles_2_show.append('training image [' + str(r) + ']')  

    show_images(images_2_show, titles_2_show)

    
    np.random.seed(1)
    tf.random.set_seed(1)
    batch_size = 128
    epochs = 10
    learning_rate = 1e-2
    intermediate_dim = 64
    original_dim = 784

    X_train = Load_Mnist_Images(train_file)
    X_train = X_train / np.max(X_train)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
    X_train = X_train.astype('float32')



def encoder(input image):


def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1    
    for x in zip(images, title_texts):        
        image = x[0]        
        title_text = x[1]
        plt.subplot(rows, cols, index)        
        plt.imshow(image, cmap=plt.cm.gray)
        if title_text != '':
            plt.title(title_text, fontsize = 15);        
        index += 1
    plt.savefig('images.jpg')

def Load_Mnist_Images(train_images_path):
    with open(train_images_path, 'rb') as file:
        magic_num, num_of_images, rows, cols = struct.unpack(">IIII", file.read(16))
        image_data = array("B", file.read())        
    images = []
    for i in range(num_of_images):
        images.append([0] * rows * cols)
    for i in range(num_of_images):
        img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
        img = img.reshape(28, 28)
        images[i][:] = img            
    print("Number of images: ",num_of_images)
    print("Rows of images: ",rows)
    print("Cols of images: ",cols)
    return images

if __name__ == "__main__":
    main(sys.argv[0:])