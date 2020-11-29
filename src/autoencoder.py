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
# from keras import layers, optimizers, losses, metrics
# import keras
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

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
    
    # np.random.seed(1)
    # tf.random.set_seed(1)
    # batch_size = 128
    # epochs = 10
    # learning_rate = 1e-2
    # intermediate_dim = 64
    # original_dim = 784

    train_data = images 

    #Rescale the training data with the maximum pixel value of them
    train_data = train_data / np.max(train_data)
    # print(train_data.shape[0])
    # print(train_data.shape[1])
    # print(train_data.shape[2])
    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2])
    train_data = train_data.astype('float32')

    # print(type(train_data))
    print("Training set (images) shape: {shape}".format(shape=train_data.shape))

    train_data = train_data.reshape(-1, train_data.shape[1],train_data.shape[1], 1)

    print(train_data.shape)

    print(train_data.dtype)

    print(np.max(train_data))

    X_train, X_val, ground_train, ground_val = train_test_split(train_data, train_data, test_size=0.2, random_state=13)

    batch_size = 128
    epochs = 50
    inChannel = 1
    x, y = 28, 28
    input_img = Input(shape = (x, y, inChannel))

    autoencoder = Model(input_img, AutoEncoder(input_img))
    autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())   

    print(autoencoder.summary())

    autoencoder_train = autoencoder.fit(X_train, ground_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_train, ground_train))

    loss = autoencoder_train.history['loss']
    val_loss = autoencoder_train.history['val_loss']
    epochs = range(epochs)
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss', color='r')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    # plt.show()
    plt.savefig('train_val_loss.jpg')
        

def AutoEncoder(input_image):
    convolution_layer = Encoder(input_image)
    decoded_layer = Decoder(convolution_layer)
    return decoded_layer

def Encoder(input_image):
    #input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_image)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3) #7
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    
    return conv4

def Decoder(conv4):
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5) #7 x 7 x 64
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    up1 = UpSampling2D((2,2))(conv6) #14 x 14 x 64
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 32
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    up2 = UpSampling2D((2,2))(conv7) # 28 x 28 x 32
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    
    return decoded


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
    # plt.savefig('images.jpg')

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