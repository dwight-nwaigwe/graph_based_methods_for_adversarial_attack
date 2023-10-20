# -*- coding: utf-8 -*-

import os
import sys
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
import networkx as nx
sys.path.insert(1, '/home/nwaigwed/scripts')
import lrp_mnist3 as lrp_mnist
import pickle
from natsort import realsorted, ns
from natsort import natsorted
from load_model_attack import *

batch_size=50
num_classes=2 #true or false
max_files_to_read=10000
train_frac=0.7
statistic=sys.argv[1] 
files_to_delete=[]
pkl_files=[]
direc='//silenus//PROJECTS//pr-deepneuro//nwaigwed//'+ model_name +'/'+ attack + '/output0.01//'
files=os.listdir(direc)

files_to_delete2=[]
pkl_files2=[]
direc2='//silenus//PROJECTS//pr-deepneuro//nwaigwed//' + model_name + '/' +  attack+'/output0.01_pure_benign/'
files2=os.listdir(direc2)

for i in range(0,len(files)):
    if 'pkl' in files[i]:
        files_to_delete.append(i)
for i in range(len(files_to_delete)-1,-1,-1 ):
    pkl_files.append(files[files_to_delete[i]])
    del files[files_to_delete[i]]


for i in range(0,len(files2)):
    if 'pkl' in files2[i]:
        files_to_delete2.append(i)
for i in range(len(files_to_delete2)-1,-1,-1 ):
    pkl_files2.append(files2[files_to_delete2[i]])
    del files2[files_to_delete2[i]]



files_benign=[file for file in files if 'adv' not in file]
files_benign=natsorted(files_benign, alg=ns.REAL)
files_adv=[file for file in files if 'adv' in file]
files_adv=natsorted(files_adv, alg=ns.REAL)
pkl_files=[file for file in pkl_files if statistic in file]
pkl_files=natsorted(pkl_files, alg=ns.REAL)

files_benign2=[file for file in files2 ]
files_benign2=natsorted(files_benign2, alg=ns.REAL)
pkl_files2=[file for file in pkl_files2 if statistic in file]
pkl_files2=natsorted(pkl_files2, alg=ns.REAL)

files_benign=files_benign[0:max_files_to_read]
files_adv=files_adv[0:max_files_to_read]
pkl_files=pkl_files[0:max_files_to_read]

files_benign2=files_benign2[0:max_files_to_read]
pkl_files2=pkl_files2[0:max_files_to_read]

res_benign=[]
for somefile in files_benign: 
    lines2=[]
    adj_benign=np.zeros([10,1010], 'float32')
    with open(direc+somefile, 'rt') as file:
        lines=file.readlines()
        for el in lines:
            tmpline=el.split(' ')
            tmpline[2]=tmpline[2][0:-1]
            lines2.append(tmpline)
    for el in lines2:
        if ((float(el[0])<10 and float(el[1])<1010)):
            adj_benign[int(el[0]),int(el[1])] =el[2]
    adj_benign_flattened=adj_benign.flatten()
    res_benign.append(adj_benign_flattened)
res_benign=np.array(res_benign)   

res_adv=[]      
for somefile in files_adv: 
    lines2=[]
    adj_adv=np.zeros([10,1010], 'float32')
    with open(direc+somefile, 'rt') as file:
        lines=file.readlines()
        for el in lines:
            tmpline=el.split(' ')
            tmpline[2]=tmpline[2][0:-1]
            lines2.append(tmpline)
    for el in lines2:
        if ((float(el[0])<10 and float(el[1])<1010)):
            adj_adv[int(el[0]),int(el[1])] =el[2]
    adj_adv_flattened=adj_adv.flatten()
    res_adv.append(adj_adv_flattened)
res_adv=np.array(res_adv)   


res_benign2=[]
for somefile in files_benign2: 
    lines2=[]
    adj_benign=np.zeros([10,1010], 'float32')
    with open(direc2+somefile, 'rt') as file:
        lines=file.readlines()
        for el in lines:
            tmpline=el.split(' ')
            tmpline[2]=tmpline[2][0:-1]
            lines2.append(tmpline)
    for el in lines2:
        if ((float(el[0])<10 and float(el[1])<1010)):
            adj_benign[int(el[0]),int(el[1])] =el[2]
    adj_benign_flattened=adj_benign.flatten()
    res_benign2.append(adj_benign_flattened)
res_benign2=np.array(res_benign2)   

stats_benign=[]
stats_adv=[]      
for somefile in pkl_files: 
    with open(direc+somefile, 'rb') as file:
        benign_stats=pickle.load(file)
        adv_stats=pickle.load(file)
    stats_benign.append([val for (key,val) in benign_stats.items()])
    stats_adv.append([val for (key,val) in adv_stats.items()])
stats_benign=np.array(stats_benign)   
stats_adv=np.array(stats_adv)   

stats_benign2=[]
for somefile in pkl_files2: 
    with open(direc2+somefile, 'rb') as file:
        benign_stats2=pickle.load(file)
    stats_benign2.append([val for (key,val) in benign_stats2.items()])
stats_benign2=np.array(stats_benign2)   



num1=int(np.floor(res_benign.shape[0]*train_frac))
num2=res_benign.shape[0]
num3=int(np.floor(res_benign2.shape[0]*train_frac))
num4=res_benign2.shape[0]

if statistic=='edges':
#create edge data
    x_train=np.concatenate((res_benign[0:num1],res_adv[0:num1]), 0)
    x_train2=res_benign2[0:num3  ]
    y_train=np.concatenate( [np.zeros(shape=(num1)), np.ones(shape=(num1))]  )
    y_train2=np.zeros(shape=(num3))

if statistic != 'edges':
    x_train=np.concatenate( (stats_benign[0:num1], stats_adv[0:num1]),0)
    x_train2= stats_benign2[0:num3]
    y_train=np.concatenate( [np.zeros(shape=(num1)), np.ones(shape=(num1))]  )
    y_train2=np.zeros(shape=(num3))

if len(x_train2) !=0:
    #combine data from pure benign set with benign/adv set
    x_train=np.concatenate( (x_train, x_train2), 0)
    y_train=np.concatenate( (y_train, y_train2))

input_shape= (x_train.shape[1])
logistic_model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Dense(1,activation="sigmoid", use_bias=False),
    ]
)


optimizer=tf.keras.optimizers.Adam(learning_rate=0.05)
logistic_model.compile(optimizer=optimizer,  loss='binary_crossentropy', metrics=['accuracy'])
checkpoint_filepath = '//silenus//PROJECTS//pr-deepneuro//nwaigwed//'+ model_name +'/'+ attack + '/' + statistic+ '/'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

history = logistic_model.fit(x_train, y_train, epochs=300, batch_size=32, validation_split=0.2, shuffle=True,callbacks=[model_checkpoint_callback] )
logistic_model.load_weights(checkpoint_filepath)









