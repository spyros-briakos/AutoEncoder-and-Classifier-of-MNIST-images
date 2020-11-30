import numpy as np
import tensorflow as tf
import math
import struct
import sys
import random
import matplotlib.pyplot as plt
from array import array
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization,Dropout
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
    train_data = images 

    #Rescale the training data with the maximum pixel value of them
    train_data = train_data / np.max(train_data)
    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2])
    train_data = train_data.astype('float32')
    print("Training set (images) shape: {shape}".format(shape=train_data.shape))
    train_data = train_data.reshape(-1, train_data.shape[1],train_data.shape[1], 1)

    # Train test split
    X_train, X_val, ground_train, ground_val = train_test_split(train_data, train_data, test_size=0.2, random_state=13)
        
    # Read hyperparameters for first time from user
    epochs = read_epochs()
    batch_size = read_batch_size()
    num_of_convlayers = read_convlayers()
    filters = read_filters(num_of_convlayers)
    filter_size = read_filter_size()
    batchordropout = dropoutorbatchnorm()
    num_of_experiment=0

    while True:

        # Check if user wants to experiment or not
        decision = int(input("\nWhat do you want? (Please enter 1 or 2 or 3)\n1)I want to experiment more!\n2)I finished my job, I want to exit!(without save)\n3)I finished my job, I want to exit!(with save)\n"))
        
        # Case: New experiment
        if decision==1:

            # Initialize variable with random value 
            hyperparameter=10
            
            while True:

                # Clear old losses (but not the first time)
                losses = []
                params= []

                # Case: User want to experiment with one specific hyperparameter
                if hyperparameter==1 or hyperparameter==2 or hyperparameter==3 or hyperparameter==4 or hyperparameter==5:
                    
                    #Initialize variable with random value 
                    experiment_decision=10

                    while True:

                        # Case: Try another value of specific hyperparameter
                        if experiment_decision==1:
                            
                            if hyperparameter==1:
                                epochs = read_epochs()
                            elif hyperparameter==2:
                                batch_size = read_batch_size()
                            elif hyperparameter==3:
                                num_of_convlayers = read_convlayers()
                                filters = read_filters(num_of_convlayers)
                            elif hyperparameter==4:
                                filters = read_filters(num_of_convlayers)
                            elif hyperparameter==5:
                                filter_size = read_filter_size()

                            temp_losses, temp_model, hyperparams = train_model(num_of_convlayers,filters,filter_size,batch_size,epochs,batchordropout,X_train,ground_train)
                            losses.append(temp_losses)
                            params.append(hyperparams)
                            num_of_experiment+=1
                            experiment_decision=10

                            
                        # Case: Plot experiment's results
                        elif experiment_decision==2:
                            plot_results(losses,params,hyperparameter,num_of_experiment)
                            experiment_decision=10
                        
                        # Case: Save model
                        elif experiment_decision==3:
                            path = input("Please give me the path to save your model: ")
                            temp_model.save(path)
                            experiment_decision=10
                        
                        # Case: Exit
                        elif experiment_decision==4:
                            hyperparameter=10
                            break
                        
                        # Store user's decision
                        else:
                            experiment_decision = int(input("\nWhat do you want? (Please enter 1 or 2 or 3 or 4)\n1)Give a different value of hyperparameter!\n2)Plot results of my experiments!\n3)Save my model!\n4)Exit!\n"))

                elif hyperparameter==6:
                    break
                # Store which hyperparameter user wants to change.      
                else: 
                    hyperparameter = int(input("\nWhich hyperparameter do you wish to change? (Please enter 1 or 2 or 3 or 4 or 5 or 6)\n1)Epochs.\n2)Batch size.\n3)Number of convolutional layers (of encoder).\n4)Filters in each convolutional layer.\n5)Filter's size.\n6)None.\n"))
            
        # Case: Exit
        elif decision==2:
            print("Bye!")
            break
        # Case: Exit
        elif decision==3:
            path = input("Please give me the path to save your model: ")
            temp_model.save(path)
            print("Bye!")
            break
#####################################################################################################################################  
      
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
    return images

class AutoEncoder():

    def __init__(self, input_image, num_of_convlayers, num_of_filters, size_of_filters, dropoutorbatch):
        self.input_image = input_image
        self.num_of_convlayers = num_of_convlayers
        self.num_of_filters = num_of_filters
        self.size_of_filters = size_of_filters
        self.dropoutorbatch = dropoutorbatch
  
    def Encoder(self):
        conv = self.input_image
        
        for i in range(self.num_of_convlayers):
        
            # Add Convolutional layer
            conv = Conv2D(self.num_of_filters[i], self.size_of_filters, activation='relu', padding='same')(conv)

            # Add Dropout or Batch_Normalization layer
            if ((i == (self.num_of_convlayers-1)) and (i >= 2)):
                
                # Case: Dropout
                if self.dropoutorbatch == 1:
                    conv = Dropout(0.25, name='encoder')(conv)

                # Case: BatchNormalization
                else:
                    conv = BatchNormalization(name='encoder')(conv)

            else: 

                # Case: Dropout
                if self.dropoutorbatch == 1:
                    conv = Dropout(0.25)(conv)
                
                # Case: BatchNormalization
                else:
                    conv = BatchNormalization()(conv)
            
            # Add MaxPooling layer
            if (i < 2):
                if (i == (self.num_of_convlayers-1)):
                    conv = MaxPooling2D(pool_size=(2, 2), name='encoder')(conv) 
                else:
                    conv = MaxPooling2D(pool_size=(2, 2))(conv) 
            
        return conv

    def Decoder(self,convolution_layer):
        conv = convolution_layer

        for i in range(self.num_of_convlayers-1):
        
            # Add Convolutional layer
            conv = Conv2D(self.num_of_filters[self.num_of_convlayers-(i+1)], self.size_of_filters, activation='relu', padding='same')(conv)
            
            # Add Dropout or Batch_Normalization layer
            if self.dropoutorbatch == 1:
                conv = Dropout(0.25)(conv)
            else:
                conv = BatchNormalization()(conv)
            
            # Add UpSampling layer
            if (i >= ((self.num_of_convlayers-1)-2)):
                conv = UpSampling2D((2,2))(conv)

        if (self.num_of_convlayers == 1 or self.num_of_convlayers == 2):
            conv = UpSampling2D((2,2))(conv)

        # Final Sigmoid layer
        conv = Conv2D(1, self.size_of_filters, activation='sigmoid', padding='same')(conv)
        
        return conv  

    def call(self):
        convolution_layer = self.Encoder()
        recostructed_layer = self.Decoder(convolution_layer)
        return recostructed_layer

# Function which stacks hyperparameters into a list and returns it
def stack_hyperparams(num_of_convlayers_,filters_,filter_size_,batch_size_,epochs_):
    stacked = []
    stacked.append(num_of_convlayers_)
    stacked.append(filters_)
    stacked.append(filter_size_)
    stacked.append(batch_size_)
    stacked.append(epochs_)
    return stacked

# Function which train a model with specific parameters
def train_model(num_of_convlayers,filters,filter_size,batch_size,epochs,dropoutorbatch,X_train,ground_train):
  
    # Shape of each photo
    input_img = Input(shape = (28, 28, 1))
    
    # Clear previous model
    tf.keras.backend.clear_session()
    
    # Define our instance of class AutoEncoder
    autoenc = AutoEncoder(input_img,num_of_convlayers,filters,filter_size,dropoutorbatch)
    
    # Define our model
    autoencoder = Model(input_img, autoenc.call())
    
    # Compile our model
    autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())  
    print(autoencoder.summary()) 
    # Train our model
    autoencoder_train = autoencoder.fit(X_train, ground_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_train, ground_train))

    train_loss = autoencoder_train.history['loss']
    val_loss = autoencoder_train.history['val_loss']
    hyperparams = stack_hyperparams(num_of_convlayers,filters,filter_size,batch_size,epochs)

    return ((train_loss,val_loss),autoencoder,hyperparams)

# Function which plots loss vs epochs for all experiments.
def plot_results(experiment_losses,experiment_hyperparams,changed_hyperparam,num_of_experiment):
    fig,ax = plt.subplots(figsize=(10,10))
    ax.set_title('Training and validation loss',fontsize=18)
    ax.set_xlabel('Epochs', fontsize=18)
    ax.set_ylabel('Loss', fontsize=18)

    num_of_experiments = len(experiment_losses)
    y_pos = np.arange(len(experiment_losses))
    values_of_changed_hyperparam = []
    train_losses = []
    val_losses = []

    for i in range(num_of_experiments):
        if changed_hyperparam==1:
            values_of_changed_hyperparam.append("Epochs:"+str(experiment_hyperparams[i][4]))
        elif changed_hyperparam==2:
            values_of_changed_hyperparam.append("Batch Size:"+str(experiment_hyperparams[i][3]))
        elif changed_hyperparam==3:
            values_of_changed_hyperparam.append("Conv Layers:"+str(experiment_hyperparams[i][0]))
        elif changed_hyperparam==4:
            values_of_changed_hyperparam.append("Filters:"+str(experiment_hyperparams[i][1]))
        elif changed_hyperparam==5:
            values_of_changed_hyperparam.append("Filter's size:"+str(experiment_hyperparams[i][2]))
        
        num_of_epochs = experiment_hyperparams[i][4]
        train_losses.append(experiment_losses[i][0][num_of_epochs-1])
        val_losses.append(experiment_losses[i][1][num_of_epochs-1])

    _=ax.plot(values_of_changed_hyperparam,train_losses, 'b', label='train', color='r')
    _=ax.plot(values_of_changed_hyperparam,val_losses, 'b', label='val')
    _=ax.legend(fontsize=14)
    
    path = "images/autoencoder/losses" + str(num_of_experiment) + ".jpg" 
    fig.savefig(path)

# Functions which read hyperparameters
def read_epochs():
    return int(input("\nPlease give me number of epochs (i.e. 10 or 20 or 30 etc.): "))

def read_batch_size():
    return int(input("\nPlease give me batch size (i.e. 32 or 64 or 128 etc.): "))

def read_convlayers():
    return int(input("\nPlease give me number of convolutional layers for Encoder (it will be the same for Decoder, i.e. 3 or 4 or 5 etc.): "))

def read_filters(num_of_convlayers):
    while True:
        filters = input("\nPlease give me list with number of filters in each convolutional layer, seperated by comma (i.e. 16,32,64): ")
        filters = list(map(int,filters.split(',')))
        if len(filters)!=num_of_convlayers:
            print("You must give a list of numbers with size as much as number of convulotional layers!\n") 
        else:
            break
    return filters

def read_filter_size():
    filter_size = input("\nPlease give me size of filter, 2 integers seperated by comma (i.e. 2,2 or 3,3 etc.): ")
    filter_size = tuple(map(int, filter_size.split(','))) 
    return filter_size

def dropoutorbatchnorm():
    return int(input("\nDropout(1) or Batch Normalization(2): "))

if __name__ == "__main__":
    main(sys.argv[0:])