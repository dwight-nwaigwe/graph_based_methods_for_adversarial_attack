# -*- coding: utf-8 -*-

import os
import sys
import tensorflow as tf
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
import math
from load_process_mnist_data_adv_create_data import *

model=tf.keras.models.load_model('/home/nwaigwed//models//' + model_name+ '.h5',compile=False)
model.layers[-1].activation=tf.nn.softmax



def create_networkx_graph(rel_weights, threshold):
    G = nx.DiGraph()
    counter=0
    for j in range(0, len(rel_weights)):
        for i in range(0,rel_weights[j].shape[1]):
            G.add_node(counter, layer=j, nodes_layer= rel_weights[j].shape[1])
            counter =counter+1
    if j==len(rel_weights)-1:
        G.add_nodes_from(range(counter, rel_weights[j].shape[0] +counter), layer=j, nodes_layer= rel_weights[j].shape[0])

    rw=rel_weights[0]
    rw=tf.reshape(rw,[-1])
    rw=rw.numpy()
    rw2=rw

    rw=rel_weights[1]
    rw=tf.reshape(rw,[-1])
    rw=rw.numpy()
    rw2=np.concatenate((rw,rw2))

    rw=rel_weights[2]
    rw=tf.reshape(rw,[-1])
    rw=rw.numpy()
    rw2=np.concatenate((rw,rw2))

    rw=rel_weights[3]
    rw=tf.reshape(rw,[-1])
    rw=rw.numpy()
    rw2=np.concatenate((rw,rw2))

    rw=rel_weights[4]
    rw=tf.reshape(rw,[-1])
    rw=rw.numpy()
    rw2=np.concatenate((rw,rw2))

    rw=rel_weights[5]
    rw=tf.reshape(rw,[-1])
    rw=rw.numpy()
    rw2=np.concatenate((rw,rw2))


    rw2.sort()
    rw2=rw2[::-1]
    indx=math.floor(len(rw2)*threshold)
    cutoff=rw2[indx]

    total=0
    for j in range(0, len(rel_weights)):
        for k in range(0,rel_weights[j].shape[0]):
            for i in range(0,rel_weights[j].shape[1]):
                if rel_weights[j][k ][i] > cutoff:
                    G.add_edge( i+total, k+ total +rel_weights[j].shape[1], weight=rel_weights[j][k][i])
        total = total+rel_weights[j].shape[1]

    return G
    


def modified_cc(G):
    G = G.reverse() 
    mod_cc=dict()
    for node in G.nodes:
        sp=nx.single_source_bellman_ford_path_length(G,node, weight='weight')
        if len(sp.keys()) ==1:
            mod_cc[node]=0
        else:
            mod_cc[node]=sum(map(lambda x: 1/2**x,sp.values()))
    return mod_cc
  


def construct_rel_weights(x_sample):
    rel_weights=[]

    relevances=lrp.lrp_runner(x_sample)
    rel_weights=relevances[1]

    inds_to_delete=[]
    for a in range(0,len(rel_weights)):
        rel_weights[a]=np.squeeze(rel_weights[a])
    return rel_weights


def compute_metric( normalize_for_degree=False):
    res_benign=dict()
    res_adv=dict()
    rel_weights=None
    
    for sample_index in range(int(sys.argv[1]),int(sys.argv[2])):
        
        x_benign_sample=x_test[sample_index]
        x_benign_sample=np.expand_dims(x_benign_sample, 0)
        y_orig=tf.math.argmax(model(x_benign_sample),axis=1)
        y_orig=y_orig.numpy()[0]
        x_adv_sample=x_test_adv[sample_index]
        x_adv_sample=np.expand_dims(x_adv_sample, 0)
        y_adv=tf.math.argmax(model(x_adv_sample),axis=1)
        y_adv=y_adv.numpy()[0]

        if y_test[sample_index] ==y_orig and  y_orig !=y_adv:

            rel_weights=construct_rel_weights(x_benign_sample) 
            G= create_networkx_graph( rel_weights,threshold)
            fh= open(outpath+'//adjlist_'+str(threshold)+'_'+str(sample_index), "wb")  
            nx.write_edgelist(G, fh, data=["weight"])
            fh.close()

            rel_weights2=construct_rel_weights(x_adv_sample) 
            G2= create_networkx_graph(rel_weights2,threshold)
            fh2 = open(outpath+'//adjlist_'+"adv"+str(threshold)+'_'+str(sample_index), "wb")  
            nx.write_edgelist(G2, fh2, data=["weight"])
            fh2.close()

            for metric in metric_list:
                if metric=='modified_cc':
                    res=modified_cc(G)
                    res2=modified_cc(G2)
                 
                if metric=='cc':
                    res=nx.closeness_centrality(G)
                    res2=nx.closeness_centrality(G2)

                if metric=='out' or metric=='outin':
                    res=G.out_degree()
                    res2=G2.out_degree()
                    res={el[0]:el[1] for el in res}
                    res2={el[0]:el[1] for el in res2}
                    if metric== 'outin':
                        res_prime=G.in_degree()
                        res2_prime=G2.in_degree()
                        res_prime={el[0]:el[1] for el in res_prime}
                        res2_prime={el[0]:el[1] for el in res2_prime}
                        res={el[0]: el[1]-res_prime[el[0]] for el in res.items()}
                        res2={el[0]: el[1]-res2_prime[el[0]] for el in res2.items()}

                maxel=max(res.keys())
                for counter in range(0, 28*28):
                    del res[maxel-counter]
              
                with open(outpath+'benign_adv_stats'+str(threshold)+metric+str(sample_index)+'.pkl', 'wb') as file:
                    pickle.dump(res, file)
                    pickle.dump(res2, file) 
 
	#we separately treat the case in which the adversarial example is classified correctly 
        elif y_test[sample_index] ==y_orig and  y_orig ==y_adv:
            rel_weights=construct_rel_weights(x_benign_sample)
            G= create_networkx_graph( rel_weights,threshold)
            fh= open(outpath2+'//adjlist_'+str(threshold)+'_'+str(sample_index), "wb")
            nx.write_edgelist(G, fh, data=["weight"])
            fh.close()

            for metric in metric_list:
                if metric=='modified_cc':
                    res=modified_cc(G)

                if metric=='cc':
                    res=nx.closeness_centrality(G)

                if metric=='out' or metric=='outin':
                    res=G.out_degree()
                    res={el[0]:el[1] for el in res}
                    if metric== 'outin':
                        res_prime=G.in_degree()
                        res_prime={el[0]:el[1] for el in res_prime}
                        res={el[0]: el[1]-res_prime[el[0]] for el in res.items()}

                maxel=max(res.keys())
                for counter in range(0, 28*28):
                    del res[maxel-counter]

                with open(outpath2+'benign_stats'+str(threshold)+metric+str(sample_index)+'.pkl', 'wb') as file:
                    pickle.dump(res, file)



lrp_formula=1
lrp = lrp_mnist.LayerwiseRelevancePropagation(model=model, lrp_formula=lrp_formula)
metric_list=['modified_cc',  'outin']
threshold=0.01
outpath='/silenus/PROJECTS/pr-deepneuro/nwaigwed/'+ '/output'+str(threshold)+'/'
outpath2='/silenus/PROJECTS/pr-deepneuro/nwaigwed/'+  '/output'+str(threshold)+ '_pure_benign/'
compute_metric(normalize_for_degree=False)





    
