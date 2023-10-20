# -*- coding: utf-8 -*-

import sys
import os
import tensorflow as tf
from tensorflow import  keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
import numpy as np
import pickle
from art.attacks.evasion import SquareAttack, BasicIterativeMethod, DeepFool, ProjectedGradientDescent,FastGradientMethod,CarliniL2Method
from art.estimators.classification import TensorFlowV2Classifier
from art.utils import load_mnist
import cv2 
from functools import partial

# Step 1: Load the MNIST dataset


dataset='cifar10'
normalize=False
if dataset=='mnist':
    input_shape = (28,28,1)
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    num_classes=10
if dataset=='cifar10':
    input_shape = (224,224,3)
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    num_classes=10
if dataset=='cifar100':
    num_classes=100
    (x_train, y_train), (x_test, y_test) =tf.keras.datasets.cifar100.load_data(label_mode="fine")

start_ind=int(sys.argv[1])*1000
end_ind=(int(sys.argv[1])+1)*1000

g=partial(cv2.resize, dsize=(224,224), interpolation=cv2.INTER_CUBIC)
x_test=x_test[start_ind:end_ind]
y_test=y_test[start_ind:end_ind]
x_test=map(g, x_test)
x_test=list(x_test)
x_test=np.asarray(x_test)
y_test=tf.one_hot(tf.squeeze(y_test), 10)

model=tf.keras.models.load_model('/silenus/PROJECTS/pr-deepneuro/nwaigwed/models/mobilenet.h5',compile=False)
model.layers[-1].activation=tf.keras.activations.linear
x_test =   tf.keras.applications.mobilenet.preprocess_input(x_test)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

# Step 3: Create the ART classifier

classifier = TensorFlowV2Classifier(
    model=model,
    loss_object=loss_object,
    nb_classes=num_classes,
    input_shape=input_shape,
    clip_values=(-1, 1),
)


# Step 5: Evaluate the ART classifier on benign test examples

predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

# Step 6: Generate adversarial test examples

#attack = FastGradientMethod(estimator=classifier, eps=0.2)
#attack = BasicIterativeMethod(estimator=classifier, eps=0.1, max_iter=10)
#attack = ProjectedGradientDescent(estimator=classifier, norm=np.inf, eps=100, eps_step=30, max_iter=10)
attack = CarliniL2Method(classifier)
#attack=DeepFool(classifier=classifier)
#attack = SquareAttack(estimator=classifier, norm=np.inf, eps=0.1)

x_test_adv = attack.generate(x=x_test)

# Step 7: Evaluate the ART classifier on adversarial test examples

predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))

with open('mobilenet_adv_examples_carlinil2_'+str(start_ind)+'.pkl', 'wb') as file:
    pickle.dump(x_test_adv, file)


