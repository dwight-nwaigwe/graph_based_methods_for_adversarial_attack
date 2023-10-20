# -*- coding: utf-8 -*-


import tensorflow as tf
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from art.utils import load_mnist
import time
import pickle

model_name='model1'
#model_attack='model1_projected'
path_adv_examples='/silenus/PROJECTS/pr-deepneuro/nwaigwed/models/'

with open(path_adv_examples+'//adv_examples_projected.pkl', 'rb') as f:
    x_test_adv = pickle.load(f)


normalize= True  #to normalize input images

num_classes=10
input_shape = (28,28,1)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#normalize or not normalize the data
if normalize is True:
    x_train=x_train/255
    x_test=x_test/255


x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]


x_train = np.expand_dims(x_train, -1)
x_val = np.expand_dims(x_val, -1)
x_test = np.expand_dims(x_test, -1)

x_train = x_train.astype("float32") 
x_test = x_test.astype("float32") 
x_val = x_val.astype("float32") 

