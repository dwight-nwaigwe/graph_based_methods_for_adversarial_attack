# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 17:41:05 2022

@author: valentin



The script demonstrates a simple example of using ART with TensorFlow v1.x. The example train a small model on the MNIST
dataset and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train
the model, it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""

import tensorflow as tf
from tensorflow import  keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
import numpy as np
import pickle
from art.attacks.evasion import AutoAttack, SquareAttack,SaliencyMapMethod, BasicIterativeMethod,DeepFool, ProjectedGradientDescent,FastGradientMethod,CarliniL2Method
from art.estimators.classification import TensorFlowV2Classifier
from art.utils import load_mnist

# Step 1: Load the MNIST dataset

#(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
dataset='mnist'
if dataset=='mnist':
    input_shape = (28,28,1)
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    num_classes=10
if dataset=='cifar10':
    input_shape = (32,32,3)
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    num_classes=10
if dataset=='cifar100':
    num_classes=100
    (x_train, y_train), (x_test, y_test) =tf.keras.datasets.cifar100.load_data(label_mode="fine")


x_test=np.expand_dims(x_test,-1)

#normalize or not normalize the data
x_test=x_test/255
#x_test=x_test[0:20]
#y_test=y_test[0:20]

#y_train2=np.zeros((len(y_train),num_classes), 'uint8')
y_test=tf.one_hot(tf.squeeze(y_test), 10)

model=tf.keras.models.load_model('../models/model1.h5',compile=False)
model.layers[-1].activation=tf.keras.activations.linear
# class TensorFlowModel(Model):
#     """
#     Standard TensorFlow model for unit testing.
#     """

#     def __init__(self):
#         super(TensorFlowModel, self).__init__()
#         self.conv1 = Conv2D(filters=4, kernel_size=5, activation="relu")
#         self.conv2 = Conv2D(filters=10, kernel_size=5, activation="relu")
#         self.maxpool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid", data_format=None)
#         self.flatten = Flatten()
#         self.dense1 = Dense(100, activation="relu")
#         self.logits = Dense(10, activation="linear")

#     def call(self, x):
#         """
#         Call function to evaluate the model.
#         :param x: Input to the model
#         :return: Prediction of the model
#         """
#         x = self.conv1(x)
#         x = self.maxpool(x)
#         x = self.conv2(x)
#         x = self.maxpool(x)
#         x = self.flatten(x)
#         x = self.dense1(x)
#         x = self.logits(x)
#         return x


optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

# Step 3: Create the ART classifier

classifier = TensorFlowV2Classifier(
    model=model,
    loss_object=loss_object,
    nb_classes=num_classes,
    input_shape=input_shape,
    clip_values=(0, 1),
)


# Step 5: Evaluate the ART classifier on benign test examples

predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

# Step 6: Generate adversarial test examples
attack = FastGradientMethod(estimator=classifier, eps=0.1)
#attack = BasicIterativeMethod(estimator=classifier, eps=.05, eps_step=.01255, max_iter=40)
#attack = ProjectedGradientDescent(estimator=classifier, norm=2, eps=1, eps_step=.05, max_iter=40)
#attack = CarliniL2Method(classifier)
#attack = SquareAttack(estimator=classifier, norm=np.inf, eps=0.05)
#attack = AutoAttack(estimator=classifier, eps=0.05)
#attack=DeepFool(classifier=classifier)
#attack=SaliencyMapMethod(classifier=classifier,theta=0.1, gamma=.14)

x_test_adv = attack.generate(x=x_test)
x_test_adv=np.squeeze(x_test_adv)


# Step 7: Evaluate the ART classifier on adversarial test examples

predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))

with open('/silenus/PROJECTS/pr-deepneuro/nwaigwed/model1_adv_examples_fgm.pkl', 'wb') as file:
    pickle.dump(x_test_adv, file)
