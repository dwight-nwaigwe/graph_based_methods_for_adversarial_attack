import sys
import os
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
import lrp_mnist3 as lrp_mnist
import pickle
import math
ssl._create_default_https_context = ssl._create_unverified_context
from load_process_mnist_data_adv import *
from scipy.stats import wasserstein_distance
import functools

model=tf.keras.models.load_model('/home/nwaigwed//models//'+model_name+'.h5',compile=False)
model.layers[-1].activation=tf.nn.softmax

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
    


def construct_rel_weights(x_sample):
    rel_weights=[]

    relevances=lrp.lrp_runner(x_sample)
    rel_weights=relevances[1]

    inds_to_delete=[]
    for a in range(0,len(rel_weights)):
        rel_weights[a]=np.squeeze(rel_weights[a])
    return rel_weights

def get_data(sample_range,chosen_class):
    benign_res=[]
    adv_res=[]
    benign_res_x=[]
    adv_res_x=[]
    adv_res_act=[]
    benign_res_act=[]
    
    
    for sample_index in sample_range:
        res=dict()
        print(sample_index)
        
        #if y_test[sample_index]==chosen_class:
        
        x_sample_orig=x_test[sample_index]
        x_sample_orig=np.expand_dims(x_sample_orig, 0)
        y_model_orig=tf.math.argmax(model(x_sample_orig),axis=1)
        y_model_orig=y_model_orig.numpy()[0]
        
        x_sample_adv=x_test_adv[sample_index]
        x_sample_adv=np.expand_dims(x_sample_adv, 0)
        y_model_adv=tf.math.argmax(model(x_sample_adv),axis=1)
        y_model_adv=y_model_adv.numpy()[0]
        
        #model_correct=y_model_orig ==chosen_class and  y_test[sample_index]==chosen_class
        #adv_check= y_model_orig !=y_model_adv and  y_test[sample_index]==y_model_orig and y_model_adv ==chosen_class
        
        model_correct=y_model_orig == y_test[sample_index] and  y_test[sample_index]==chosen_class
        adv_check= y_model_orig !=y_model_adv and  y_test[sample_index]==y_model_orig  and y_model_adv==chosen_class
        
        if model_correct:
            rel_weights=construct_rel_weights(x_sample_orig)
            activs=lrp.lrp_runner_activations(x_sample_orig)
            if model_name=='model1':
                activs.pop(0)
            activs.reverse()
            G= create_networkx_graph( rel_weights,threshold)
            
            if metric !='edges':
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
                for counter in range(0, input_shape[0]* input_shape[1]* input_shape[2]):
                    del res[maxel-counter]
            
                if metric=='outin' or metric=='modified_cc':
                    res=[el[1] for el in res.items()]
                    #res=np.expand_dims(res,0)
                    
            if metric=='edges':
                lines2=[]
                adj_matrix=np.zeros([10,110], 'float32')
                for line in nx.generate_edgelist(G, data=True):
                    splitted_line=line.split(' ')
                    splitted_line[2]=splitted_line[3][:-1]
                    splitted_line=splitted_line[0:-1]
                    lines2.append(splitted_line)
                for el in lines2:
                    if ((float(el[0])<10 and float(el[1])<110)):
                        adj_matrix[int(el[0]),int(el[1])] =el[2]
                adj_matrix_flattened=adj_matrix.flatten()
                res=adj_matrix_flattened
                #adj_matrix_flattened=np.expand_dims(adj_matrix_flattened,0)

            benign_res.append( res)
            benign_res_x.append( layers.Flatten()(x_sample_orig))
            benign_res_act.append(activs)

        if adv_check:
            rel_weights2=construct_rel_weights(x_sample_adv)
            activs2=lrp.lrp_runner_activations(x_sample_adv)
            if model_name=='model1':
                activs2.pop(0)
            activs2.reverse()
            G2= create_networkx_graph( rel_weights2,threshold)
            
            if metric != 'edges':
                if metric=='modified_cc':
                    res2=modified_cc(G2)
                        
                if metric=='cc':
                    res2=nx.closeness_centrality(G2)
            
                if metric=='out' or metric=='outin':
                    res2=G2.out_degree()
                    res2={el[0]:el[1] for el in res2}
                    if metric== 'outin':
                        res2_prime=G2.in_degree()
                        res2_prime={el[0]:el[1] for el in res2_prime}
                        res2={el[0]: el[1]-res2_prime[el[0]] for el in res2.items()}
                maxel=max(res2.keys())
                for counter2 in range(0, input_shape[0]* input_shape[1]* input_shape[2]):
                    del res2[maxel-counter2]
                    
                if metric=='outin' or metric=='modified_cc':
                    res2=[el[1] for el in res2.items()]
                    #res=np.expand_dims(res,0)

                
            if metric=='edges':                    
                lines2=[]
                adj_matrix=np.zeros([10,110], 'float32')
                for line in nx.generate_edgelist(G2, data=True):
                    splitted_line=line.split(' ')
                    splitted_line[2]=splitted_line[3][:-1]
                    splitted_line=splitted_line[0:-1]
                    lines2.append(splitted_line)
                for el in lines2:
                    if ((float(el[0])<10 and float(el[1])<110)):
                        adj_matrix[int(el[0]),int(el[1])] =el[2]
                adj_matrix_flattened=adj_matrix.flatten()
                res2=adj_matrix_flattened

            adv_res.append( res2)    
            adv_res_x.append( layers.Flatten()(x_sample_adv))
            adv_res_act.append(activs2)
    return benign_res,  benign_res_act, benign_res_x, adv_res, adv_res_act, adv_res_x


def get_data2(x_sample, activations):
#if y_test[sample_index]==chosen_class:
    if activations==True:
        activs=lrp.lrp_runner_activations(x_sample)
        if model_name=='model1':
            activs.pop(0)
        activs.reverse()
        return activs
    
    rel_weights=construct_rel_weights(x_sample)
    G= create_networkx_graph( rel_weights,threshold)
    
    if metric !='edges':
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
        for counter in range(0, input_shape[0]* input_shape[1]* input_shape[2]):
            del res[maxel-counter]
    
        if metric=='outin' or metric=='modified_cc':
            res=res=[el[1] for el in res.items()]
            #res=np.expand_dims(res,0)
            
    if metric=='edges':
        lines2=[]
        adj_matrix=np.zeros([10,110], 'float32')
        for line in nx.generate_edgelist(G, data=True):
            splitted_line=line.split(' ')
            splitted_line[2]=splitted_line[3][:-1]
            splitted_line=splitted_line[0:-1]
            lines2.append(splitted_line)
        for el in lines2:
            if ((float(el[0])<10 and float(el[1])<110)):
                adj_matrix[int(el[0]),int(el[1])] =el[2]
        adj_matrix_flattened=adj_matrix.flatten()
        res=adj_matrix_flattened
        #adj_matrix_flattened=np.expand_dims(adj_matrix_flattened,0)
    return res


def test_classification(inds, activations):
    counter=0
    counter2=0
    counter_benign=0
    counter_adv=0
    dist1=[]
    dist2=[]

    for sample_index in inds:
        print(sample_index)
    
        x_sample_orig=x_test[sample_index]
        x_sample_orig=np.expand_dims(x_sample_orig, 0)
        y_model_orig=tf.math.argmax(model(x_sample_orig),axis=1)
        y_model_orig=y_model_orig.numpy()[0]
        
        x_sample_adv=x_test_adv[sample_index]
        x_sample_adv=np.expand_dims(x_sample_adv, 0)
        y_model_adv=tf.math.argmax(model(x_sample_adv),axis=1)
        y_model_adv=y_model_adv.numpy()[0]
        
        model_correct=y_model_orig == y_test[sample_index] 
        adv_check= y_model_orig !=y_model_adv and  y_test[sample_index]==y_model_orig
        
        if activations==False:
            
            if model_correct==True and test_benign==True:
                res=get_data2( x_sample_orig, activations)
                counter=counter+1
                y_model=y_model_orig
            if adv_check==True and test_benign==False:
                res=get_data2(x_sample_adv, activations)
                counter=counter+1
                y_model=y_model_adv
            
            if   (model_correct==True and test_benign==True) or (adv_check==True and test_benign==False):
                if y_model ==0:
                    benign_res=benign_res_0
                    adv_res=adv_res_0
                if y_model  ==1:
                    benign_res=benign_res_1
                    adv_res=adv_res_1
                if y_model  ==2:
                    benign_res=benign_res_2
                    adv_res=adv_res_2
                if y_model  ==3:
                    benign_res=benign_res_3
                    adv_res=adv_res_3
                if y_model  ==4:
                    benign_res=benign_res_4
                    adv_res=adv_res_4
                if y_model ==5:
                    benign_res=benign_res_5
                    adv_res=adv_res_5
                if y_model  ==6:
                    benign_res=benign_res_6
                    adv_res=adv_res_6
                if y_model  ==7:
                    benign_res=benign_res_7
                    adv_res=adv_res_7
                if y_model  ==8:
                    benign_res=benign_res_8
                    adv_res=adv_res_8
                if y_model  ==9:
                    benign_res=benign_res_9
                    adv_res=adv_res_9
            
                dist_benign=[]
                dist_adv=[]
                for k in var_inds:
                    dist_benign.append(wasserstein_distance( [res[k]],benign_res[:,k]  ))
                    dist_adv.append(wasserstein_distance([res[k]],adv_res[:,k]  ))
                    
                    
                    
                tmp1=sum(dist_benign )
                tmp2=sum(dist_adv )
                print(tmp1, tmp2)
                
                if  tmp1< cutoff: #and sum(dist_adv)<cutoff: #  sum(dist_benign)<sum(dist_adv):
                    counter_benign=counter_benign+1
    
                if  tmp2> cutoff: #and sum(dist_adv)>cutoff:  #sum(dist_benign)>sum(dist_adv) or 1.1*sum(dist_benign)>sum(dist_adv) :
                    counter_adv=counter_adv+1
                dist1.append(tmp1)
                dist2.append(tmp2)  
                    
        else:
            if model_correct==True and test_benign==True:
                res=get_data2( x_sample_orig,True)
                counter=counter+1
                y_model=y_model_orig
                
                
            if adv_check==True and test_benign==False:
                res=get_data2(x_sample_adv, True)
                counter=counter+1
                y_model=y_model_adv
                

            if   (model_correct==True and test_benign==True) or (adv_check==True and test_benign==False):
                if y_model ==0:
                    benign_res=benign_res_act_0
                    adv_res=adv_res_act_0
                if y_model  ==1:
                    benign_res=benign_res_act_1
                    adv_res=adv_res_act_1
                if y_model  ==2:
                    benign_res=benign_res_act_2
                    adv_res=adv_res_act_2
                if y_model  ==3:
                    benign_res=benign_res_act_3
                    adv_res=adv_res_act_3
                if y_model  ==4:
                    benign_res=benign_res_act_4
                    adv_res=adv_res_act_4
                if y_model ==5:
                    benign_res=benign_res_act_5
                    adv_res=adv_res_act_5
                if y_model  ==6:
                    benign_res=benign_res_act_6
                    adv_res=adv_res_act_6
                if y_model  ==7:
                    benign_res=benign_res_act_7
                    adv_res=adv_res_act_7
                if y_model  ==8:
                    benign_res=benign_res_act_8
                    adv_res=adv_res_act_8
                if y_model  ==9:
                    benign_res=benign_res_act_9
                    adv_res=adv_res_act_9
                    
                #res =  [ el.numpy() for el  in res]
                res =  [ el.flatten() for el  in res]
                res=tf.concat(res,0) 
            
                dist_benign=[]
                dist_adv=[]
                for k in var_inds:# in range (0,benign_res.shape[1]):
                    dist_benign.append(wasserstein_distance( [res[k]],benign_res[:,k]  ))
                    dist_adv.append(wasserstein_distance([res[k]],adv_res[:,k]  ))
                    
                    
                tmp1=sum(dist_benign )
                tmp2=sum(dist_adv )
                print(tmp1, tmp2)
                
                if  tmp1< cutoff: #and sum(dist_adv)<cutoff: #  sum(dist_benign)<sum(dist_adv):
                    counter_benign=counter_benign+1
    
                if  tmp2> cutoff: #and sum(dist_adv)>cutoff:  #sum(dist_benign)>sum(dist_adv) or 1.1*sum(dist_benign)>sum(dist_adv) :
                    counter_adv=counter_adv+1
                dist1.append(tmp1)
                dist2.append(tmp2)  

    print(counter_benign/counter, counter_adv/counter)
    return dist1, dist2

def test_classification_specific(inds, activations):
    counter=0
    counter2=0
    counter_benign=0
    counter_adv=0
    dist1=[]
    dist2=[]

    for sample_index in inds:
        print(sample_index)
    
        x_sample_orig=x_test[sample_index]
        x_sample_orig=np.expand_dims(x_sample_orig, 0)
        y_model_orig=tf.math.argmax(model(x_sample_orig),axis=1)
        y_model_orig=y_model_orig.numpy()[0]
        
        x_sample_adv=x_test_adv[sample_index]
        x_sample_adv=np.expand_dims(x_sample_adv, 0)
        y_model_adv=tf.math.argmax(model(x_sample_adv),axis=1)
        y_model_adv=y_model_adv.numpy()[0]
        
        model_correct=y_model_orig == y_test[sample_index] 
        adv_check= y_model_orig !=y_model_adv and  y_test[sample_index]==y_model_orig
        
        if activations==False:
            
            if model_correct==True and test_benign==True:
                res=get_data2( x_sample_orig, activations)
                counter=counter+1
                y_model=y_model_orig
            if adv_check==True and test_benign==False:
                res=get_data2(x_sample_adv, activations)
                counter=counter+1
                y_model=y_model_adv
 
            if   (model_correct==True and test_benign==True) or (adv_check==True and test_benign==False):
                if y_model ==0:
                    benign_res=benign_res_0
                    adv_res=adv_res_0
                    var_inds=[i for i in range(len(varlistdiff_0)) if varlistdiff_0[i]<0]
                if y_model  ==1:
                    benign_res=benign_res_1
                    adv_res=adv_res_1
                    var_inds=[i for i in range(len(varlistdiff_1)) if varlistdiff_1[i]<0]
                if y_model  ==2:
                    benign_res=benign_res_2
                    adv_res=adv_res_2
                    var_inds=[i for i in range(len(varlistdiff_2)) if varlistdiff_2[i]<0]
                if y_model  ==3:
                    benign_res=benign_res_3
                    adv_res=adv_res_3
                    var_inds=[i for i in range(len(varlistdiff_3)) if varlistdiff_3[i]<0]
                if y_model  ==4:
                    benign_res=benign_res_4
                    adv_res=adv_res_4
                    var_inds=[i for i in range(len(varlistdiff_4)) if varlistdiff_4[i]<0]
                if y_model ==5:
                    benign_res=benign_res_5
                    adv_res=adv_res_5
                    var_inds=[i for i in range(len(varlistdiff_5)) if varlistdiff_5[i]<0]
                if y_model  ==6:
                    benign_res=benign_res_6
                    adv_res=adv_res_6
                    var_inds=[i for i in range(len(varlistdiff_6)) if varlistdiff_6[i]<0]
                if y_model  ==7:
                    benign_res=benign_res_7
                    adv_res=adv_res_7
                    var_inds=[i for i in range(len(varlistdiff_7)) if varlistdiff_7[i]<0]
                if y_model  ==8:
                    benign_res=benign_res_8
                    adv_res=adv_res_8
                    var_inds=[i for i in range(len(varlistdiff_8)) if varlistdiff_8[i]<0]
                if y_model  ==9:
                    benign_res=benign_res_9
                    adv_res=adv_res_9
                    var_inds=[i for i in range(len(varlistdiff_9)) if varlistdiff_9[i]<0]
            
                dist_benign=[]
                dist_adv=[]
                for k in var_inds:
                    dist_benign.append(wasserstein_distance( [res[k]],benign_res[:,k]  ))
                    dist_adv.append(wasserstein_distance([res[k]],adv_res[:,k]  ))
                    
                    
                tmp1=sum(dist_benign )
                tmp2=sum(dist_adv )
                print(tmp1, tmp2)
                
                if  tmp1< cutoff: #and sum(dist_adv)<cutoff: #  sum(dist_benign)<sum(dist_adv):
                    counter_benign=counter_benign+1
    
                if  tmp2> cutoff: #and sum(dist_adv)>cutoff:  #sum(dist_benign)>sum(dist_adv) or 1.1*sum(dist_benign)>sum(dist_adv) :
                    counter_adv=counter_adv+1
                dist1.append(tmp1)
                dist2.append(tmp2)  
                    
        else:
            if model_correct==True and test_benign==True:
                res=get_data2( x_sample_orig,True)
                counter=counter+1
                y_model=y_model_orig
                
                
            if adv_check==True and test_benign==False:
                res=get_data2(x_sample_adv, True)
                counter=counter+1
                y_model=y_model_adv
                

            if   (model_correct==True and test_benign==True) or (adv_check==True and test_benign==False):

                if y_model ==0:
                    benign_res=benign_res_act_0
                    adv_res=adv_res_act_0
                    var_inds=[i for i in range(len(varlistdiff_0)) if varlistdiff_0[i]<0]

                if y_model  ==1:
                    benign_res=benign_res_act_1
                    adv_res=adv_res_act_1
                    var_inds=[i for i in range(len(varlistdiff_1)) if varlistdiff_1[i]<0]

                if y_model  ==2:
                    benign_res=benign_res_act_2
                    adv_res=adv_res_act_2
                    var_inds=[i for i in range(len(varlistdiff_2)) if varlistdiff_2[i]<0]

                if y_model  ==3:
                    benign_res=benign_res_act_3
                    adv_res=adv_res_act_3
                    var_inds=[i for i in range(len(varlistdiff_3)) if varlistdiff_3[i]<0]

                if y_model  ==4:
                    benign_res=benign_res_act_4
                    adv_res=adv_res_act_4
                    var_inds=[i for i in range(len(varlistdiff_4)) if varlistdiff_4[i]<0]

                if y_model ==5:
                    benign_res=benign_res_act_5
                    adv_res=adv_res_act_5
                    var_inds=[i for i in range(len(varlistdiff_5)) if varlistdiff_5[i]<0]

                if y_model  ==6:
                    benign_res=benign_res_act_6
                    adv_res=adv_res_act_6
                    var_inds=[i for i in range(len(varlistdiff_6)) if varlistdiff_6[i]<0]

                if y_model  ==7:
                    benign_res=benign_res_act_7
                    adv_res=adv_res_act_7
                    var_inds=[i for i in range(len(varlistdiff_7)) if varlistdiff_7[i]<0]

                if y_model  ==8:
                    benign_res=benign_res_act_8
                    adv_res=adv_res_act_8
                    var_inds=[i for i in range(len(varlistdiff_8)) if varlistdiff_8[i]<0]

                if y_model  ==9:
                    benign_res=benign_res_act_9
                    adv_res=adv_res_act_9
                    var_inds=[i for i in range(len(varlistdiff_9)) if varlistdiff_9[i]<0]              
                    
                    
                #res =  [ el.numpy() for el  in res]
                res =  [ el.flatten() for el  in res]
                res=tf.concat(res,0) 
            
                dist_benign=[]
                dist_adv=[]
                for k in var_inds:# in range (0,benign_res.shape[1]):
                    dist_benign.append(wasserstein_distance( [res[k]],benign_res[:,k]  ))
                    dist_adv.append(wasserstein_distance([res[k]],adv_res[:,k]  ))
                    
                    
                tmp1=sum(dist_benign )
                tmp2=sum(dist_adv )
                print(tmp1, tmp2)
                
                if  tmp1< cutoff: #and sum(dist_adv)<cutoff: #  sum(dist_benign)<sum(dist_adv):
                    counter_benign=counter_benign+1
    
                if  tmp2> cutoff: #and sum(dist_adv)>cutoff:  #sum(dist_benign)>sum(dist_adv) or 1.1*sum(dist_benign)>sum(dist_adv) :
                    counter_adv=counter_adv+1
                dist1.append(tmp1)
                dist2.append(tmp2)  

    print(counter_benign/counter, counter_adv/counter)
    return dist1, dist2


   

outpath='/silenus/PROJECTS/pr-deepneuro/nwaigwed/wasserstein_' + model_name + '/'
threshold=0.01
lrp_formula=1
lrp = lrp_mnist.LayerwiseRelevancePropagation(model=model, lrp_formula=lrp_formula)
metric='outin'
activations=False #for a different experiment 


benign_res, benign_res_act, benign_res_x, adv_res,adv_res_act,  adv_res_x=get_data(range(0,7000), int(sys.argv[1]) )
with open(outpath+ model_attack+'_'+ metric+'_class'+'_'+str(sys.argv[1])+'.pkl', 'wb') as file:
    pickle.dump(benign_res, file)
    pickle.dump(benign_res_act, file)
    pickle.dump( benign_res_x, file)
    pickle.dump(adv_res, file)
    pickle.dump(adv_res_act, file)
    pickle.dump(adv_res_x, file)





