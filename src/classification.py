import numpy as np
import tensorflow as tf
from tensorflow import keras
import struct
import math
from array import array
import sys
import random
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization,LeakyReLU,Dense,Dropout,Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import PIL
from tensorflow.keras.regularizers import l2

def main(argv):
    if(len(sys.argv) != 11):
        print("Sorry, the input must be in this form: classification.py  –d  <training  set>  –dl  <training  labels> -t <testset> -tl <test labels> -model <autoencoder h5>")
        exit(1)
    else:
        if(str(sys.argv[1]) != '-d'):
            print("Sorry, the input must be in this form: classification.py  –d  <training  set>  –dl  <training  labels> -t <testset> -tl <test labels> -model <autoencoder h5>")
            exit(1)
        else:
            train_file = str(sys.argv[2])

        if(str(sys.argv[3]) != '-dl'):
            print("Sorry, the input must be in this form: classification.py  –d  <training  set>  –dl  <training  labels> -t <testset> -tl <test labels> -model <autoencoder h5>")
            exit(1)
        else:
            train_labels = str(sys.argv[4]) 

        if(str(sys.argv[5]) != '-t'):
            print("Sorry, the input must be in this form: classification.py  –d  <training  set>  –dl  <training  labels> -t <testset> -tl <test labels> -model <autoencoder h5>")
            exit(1)
        else:
            test_file = str(sys.argv[6])    

        if(str(sys.argv[7]) != '-tl'):
            print("Sorry, the input must be in this form: classification.py  –d  <training  set>  –dl  <training  labels> -t <testset> -tl <test labels> -model <autoencoder h5>")
            exit(1)
        else:
            test_labels = str(sys.argv[8])  

        if(str(sys.argv[9]) != '-model'):
            print("Sorry, the input must be in this form: classification.py  –d  <training  set>  –dl  <training  labels> -t <testset> -tl <test labels> -model <autoencoder h5>")
            exit(1)
        else:
            autoencoder = str(sys.argv[10])

        # print(train_file)
        # print(train_labels)
        # print(test_file)
        # print(test_labels)
        # print(autoencoder)
    
    (train_data, train_labels) = Load_Mnist_Images(train_file,train_labels)
    (test_data, test_labels) = Load_Mnist_Images(test_file,test_labels)

    #Rescale the training data with the maximum pixel value of them
    train_data = train_data / np.max(train_data)
    test_data = test_data / np.max(test_data)

    print("\nTraining set's shape: {shape}".format(shape=train_data.shape))
    print("Test set's shape: {shape}".format(shape=test_data.shape),"\n")

    # Find the unique numbers from the train labels
    classes = np.unique(train_labels)
    num_classes = len(classes)

    plt.figure(figsize=[5,5])

    # Display the first image in training data
    plt.subplot(121)
    plt.imshow(train_data[0,:,:], cmap='gray')
    plt.title("Ground Truth : {}".format(train_labels[0]))

    # Display the first image in testing data
    plt.subplot(122)
    plt.imshow(test_data[0,:,:], cmap='gray')
    plt.title("Ground Truth : {}".format(test_labels[0]))

    # Reshape train and test data
    train_data = train_data.reshape(-1, train_data.shape[1],train_data.shape[1], 1)
    test_data = test_data.reshape(-1, test_data.shape[1],test_data.shape[1], 1)

    # Convert from float64 to float 32.
    train_data = train_data.astype('float32')
    test_data = test_data.astype('float32')

    # Change the labels from categorical to one-hot encoding
    train_labels_one_hot = to_categorical(train_labels)
    test_labels_one_hot = to_categorical(test_labels)

    # Split train dataset to train and validation datasets.
    X_train, X_val, Y_train, Y_val = train_test_split(train_data, train_labels_one_hot, test_size=0.2, random_state=13)
    print("Shape of X_train: ", X_train.shape)
    print("Shape of X_val: ", X_val.shape)
    print("Shape of Y_train: ", Y_train.shape)
    print("Shape of Y_val: ", Y_val.shape, "\n")

    # Load AutoEncoder that user give us
    autoencoder = keras.models.load_model(autoencoder)

    # Read hyperparameters for first time from user
    epochs = read_epochs()
    batch_size = read_batch_size()
    fcunits = read_fcunits()
    dropout_ = dropoutornot()
    num_of_experiments=0

    # User's Interface
    while True:

        # Check if user wants to experiment or not
        decision = int(input("\nWhat do you want? (Please enter 1 or 2 or 3)\n1)I want to experiment!\n2)I finished my job, I want to exit!(without save)\n3)I finished my job, I want to exit!(with save)\n"))
        
        # Case: New experiment
        if decision==1:

            # Initialize variable with random value 
            hyperparameter=10
            
            while True:

                losses = []
                params= []
                
                # Case: User want to experiment with one specific hyperparameter
                if hyperparameter==1 or hyperparameter==2 or hyperparameter==3:
                    
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
                                fcunits = read_fcunits()

                            temp_losses, temp_model, hyperparams = train_model(autoencoder,num_classes,fcunits,batch_size,epochs,dropout_,X_train,Y_train,X_val,Y_val)
                            losses.append(temp_losses)
                            params.append(hyperparams)
                            num_of_experiments+=1
                            experiment_decision=10

                        # Case: Plot experiment's results
                        elif experiment_decision==2:
                            plot_results(losses,params,hyperparameter,num_of_experiments)
                            experiment_decision=10
                        
                        # Case: Predict labels on test data
                        elif experiment_decision==3:
                            predict_labels(temp_model,test_data,test_labels,num_classes,num_of_experiments)
                            experiment_decision=10

                        # Case: Save model
                        elif experiment_decision==4:
                            path = input("Please give me the path to save your model: ")
                            temp_model.save(path)
                            experiment_decision=10
                        
                        # Case: Exit
                        elif experiment_decision==5:
                            hyperparameter=10
                            break
                        
                        # Store user's decision
                        else:
                            experiment_decision = int(input("\nWhat do you want? (Please enter 1 or 2 or 3 or 4 or 5)\n1)Give a different value of hyperparameter!\n2)Plot losses of my experiments!\n3)Predict labels on test data!\n4)Save my model!\n5)Exit!\n"))

                elif hyperparameter==4:
                    break
                # Store which hyperparameter user wants to change.      
                else: 
                    hyperparameter = int(input("\nWhich hyperparameter do you wish to change? (Please enter 1 or 2 or 3 or 4)\n1)Epochs.\n2)Batch size.\n3)Fully Connected layer units.\n4)None.\n"))
            
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

################################FUNCTIONS###########################################

class FC_Neural_Network():
  
  def __init__(self, autoencoder, num_of_dence_units, num_of_classes,dropoutornot_):
    self.encoder = Model(inputs=autoencoder.input,outputs=autoencoder.get_layer('encoder').output)
    self.num_of_dence_units = num_of_dence_units
    self.num_classes = num_of_classes
    self.input_img = Input(shape = (self.encoder.output_shape[1], self.encoder.output_shape[2], self.encoder.output_shape[3]))
    self.dropoutornot = dropoutornot_

  def merge_models(self):
    self.model = Sequential()
    
    for layer in self.encoder.layers:
      self.model.add(layer)

    for layer in self.model.layers:
      layer.trainable = False
    
    print("Encoder Layers:",len(self.encoder.layers))

    self.model.add(Flatten())
    self.model.add(Dense(self.num_of_dence_units, activation='relu',kernel_regularizer=l2(1e-2),bias_regularizer=l2(1e-2)))
    if self.dropoutornot == 1:
      self.model.add((Dropout(0.25,name='new')))
    self.model.add(Dense(self.num_classes, activation='softmax'))

    print("Classifier Layers:",len(self.model.layers))

    return self.model

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
    plt.show()

def Load_Mnist_Images(images_path, labels_path):
    labels = []
    with open(labels_path, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        labels = array("B", file.read()) 

    with open(images_path, 'rb') as file:
        magic_num, num_of_images, rows, cols = struct.unpack(">IIII", file.read(16))
        image_data = array("B", file.read())        
    images = []
    for i in range(num_of_images):
        images.append([0] * rows * cols)
    for i in range(num_of_images):
        img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
        img = img.reshape(28, 28)
        images[i][:] = img            

    return images, labels

# Function which stacks hyperparameters into a list and returns it
def stack_hyperparams(fc_units_,batch_size_,epochs_,dropoutornot_):
    stacked = []
    stacked.append(epochs_)
    stacked.append(batch_size_)
    stacked.append(fc_units_)
    stacked.append(dropoutornot_)
    return stacked

# Function which train a model with specific parameters
def train_model(autoencoder,num_classes,fc_units,batch_size,epochs,dropoutornot,X_train,Y_train,X_val,Y_val):
  
    # Clear previous model
    tf.keras.backend.clear_session()

    # Define our instance of class FC_NN
    model = FC_Neural_Network(autoencoder,fc_units,num_classes,dropoutornot).merge_models()

    # Compile our model
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.RMSprop(learning_rate=1e-4),metrics=['accuracy'])
    
    # Check our model
    print(model.summary())

    epochs_of_first_train = 10
    
    # Train our model
    classifier_train = model.fit(X_train, Y_train, batch_size=batch_size,epochs=epochs_of_first_train,verbose=1,validation_data=(X_val, Y_val))

    # Active the trainability of encoded layers for the second fit
    for layer in model.layers[:-4]:
        layer.trainable = True

    # Compile again the model, after we change the trainability of some layers
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.RMSprop(learning_rate=1e-4),metrics=['accuracy'])

    # Train the model for second time 
    classifier_train_1 = model.fit(X_train, Y_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_val, Y_val))

    train_loss = classifier_train.history['loss'] + classifier_train_1.history['loss']
    val_loss = classifier_train.history['val_loss'] + classifier_train_1.history['val_loss']
    hyperparams = stack_hyperparams(fc_units,batch_size,epochs,dropoutornot)

    # Plot loss vs epochs
    fig,ax = plt.subplots(1,figsize=(8,8))
    _=ax.plot(train_loss, 'r', label='Training loss')
    _=ax.plot(val_loss, 'b', label='Validation loss')
    _=ax.set_title('Training and validation loss')
    _=ax.legend()

    return ((train_loss,val_loss),model,hyperparams)

# Function which predicts labels on test data of a specific model
def predict_labels(cf_model,test_data,test_labels,num_classes,num_of_experiment):
    predicted_classes = cf_model.predict(test_data)
    predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
    test_labels = np.array(test_labels)

    correct = np.where(predicted_classes==test_labels)[0]

    fig, axes = plt.subplots(3,3,figsize=(10,10))
    fig.suptitle('Correct Labels',fontsize=24)
    for i in range(3):
        for j in range(3):
            index = i*3+j
            axes[i,j].plot(3,3,index)
            axes[i,j].imshow(test_data[correct[index]].reshape(28,28), cmap='gray', interpolation='none')
            axes[i,j].set_title("Predicted {}, Class {}".format(predicted_classes[correct[index]], test_labels[correct[index]]))
        
    path_correct = "images/classifier/labels/correct" + str(num_of_experiment) + ".jpg" 
    fig.savefig(path_correct)

    incorrect = np.where(predicted_classes!=test_labels)[0]

    fig2, axes2 = plt.subplots(3,3,figsize=(10,10))  
    fig2.suptitle('Incorrect Labels',fontsize=24)
    for i in range(3):
        for j in range(3):
            index = i*3+j
            axes2[i,j].plot(3,3,index)
            axes2[i,j].imshow(test_data[incorrect[index]].reshape(28,28), cmap='gray', interpolation='none')
            axes2[i,j].set_title("Predicted {}, Class {}".format(predicted_classes[incorrect[index]], test_labels[incorrect[index]]))
        
    path_incorrect = "images/classifier/labels/incorrect" + str(num_of_experiment) + ".jpg" 
    fig2.savefig(path_incorrect)

    print("Found %d correct labels" % len(correct))
    print("Found %d incorrect labels" % len(incorrect))
    target_names = ["Class {}".format(i) for i in range(num_classes)]
    print(classification_report(test_labels, predicted_classes, target_names=target_names))

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
            values_of_changed_hyperparam.append("Epochs:"+str(experiment_hyperparams[i][0]))
        elif changed_hyperparam==2:
            values_of_changed_hyperparam.append("Batch Size:"+str(experiment_hyperparams[i][1]))
        elif changed_hyperparam==3:
            values_of_changed_hyperparam.append("FC Units:"+str(experiment_hyperparams[i][2]))
    
        num_of_epochs = experiment_hyperparams[i][0]
        train_losses.append(experiment_losses[i][0][num_of_epochs-1])
        val_losses.append(experiment_losses[i][1][num_of_epochs-1])

    _=ax.plot(values_of_changed_hyperparam,train_losses, 'b', label='train', color='r')
    _=ax.plot(values_of_changed_hyperparam,val_losses, 'b', label='val')
    _=ax.legend(fontsize=14)

    losses_path = "images/classifier/losses/experiment_loss" + str(num_of_experiment) + ".jpg" 
    fig.savefig(losses_path)

# Functions which read hyperparameters
def read_epochs():
    return int(input("\nPlease give me number of epochs (i.e. 10 or 20 or 30 etc.): "))

def read_batch_size():
    return int(input("\nPlease give me batch size (i.e. 32 or 64 or 128 etc.): "))

def read_fcunits():
    return int(input("\nPlease give me number of units of Fully Connected layer (i.e. 128 or 256 etc): "))

def dropoutornot():
    return int(input("\nFully Connected Layer With Dropout(1) or Without Dropout(2): "))

if __name__ == "__main__":
    main(sys.argv[0:])