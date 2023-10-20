# -*- coding: utf-8 -*-


import tensorflow as tf
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import time
import ssl
import pickle
from functools import partial
import cv2

model_name='mobilenet'
attack='autoattack'



path_adv_examples='/silenus/PROJECTS/pr-deepneuro/nwaigwed/models/'

with open(path_adv_examples+'//adv_examples_autoattack_0.pkl', 'rb') as f:
    x_test_adv0 = pickle.load(f)
with open(path_adv_examples+'//adv_examples_autoattack_1000.pkl', 'rb') as f:
    x_test_adv1 = pickle.load(f)
with open(path_adv_examples+'//adv_examples_autoattack_2000.pkl', 'rb') as f:
    x_test_adv2 = pickle.load(f)
with open(path_adv_examples+'//adv_examples_autoattack_3000.pkl', 'rb') as f:
    x_test_adv3 = pickle.load(f)
with open(path_adv_examples+'//adv_examples_autoattack_4000.pkl', 'rb') as f:
    x_test_adv4 = pickle.load(f)
with open(path_adv_examples+'//adv_examples_autoattack_5000.pkl', 'rb') as f:
    x_test_adv5 = pickle.load(f)
with open(path_adv_examples+'//adv_examples_autoattack_6000.pkl', 'rb') as f:
    x_test_adv6 = pickle.load(f)
with open(path_adv_examples+'//adv_examples_autoattack_7000.pkl', 'rb') as f:
    x_test_adv7 = pickle.load(f)
with open(path_adv_examples+'//adv_examples_autoattack_8000.pkl', 'rb') as f:
    x_test_adv8 = pickle.load(f)
with open(path_adv_examples+'//adv_examples_autoattack_9000.pkl', 'rb') as f:
    x_test_adv9 = pickle.load(f)
x_test_adv=np.concatenate( (x_test_adv0,x_test_adv1,x_test_adv2,x_test_adv3, x_test_adv4, x_test_adv5, x_test_adv6, x_test_adv7, x_test_adv8, x_test_adv9), 0)






num_classes=10
input_shape = (224,224,3)
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_test=np.squeeze(y_test)
y_train=np.squeeze(y_train)


x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]


x_train = x_train.astype("float32") 
x_test = x_test.astype("float32") 
x_val = x_val.astype("float32") 


