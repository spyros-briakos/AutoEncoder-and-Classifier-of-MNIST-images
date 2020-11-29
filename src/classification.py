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

        print(train_file)
        print(train_labels)
        print(test_file)
        print(test_labels)
        print(autoencoder)


if __name__ == "__main__":
    main(sys.argv[0:])
