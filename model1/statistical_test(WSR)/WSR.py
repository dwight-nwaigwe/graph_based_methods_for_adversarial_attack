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
from sklearn import metrics



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
    


#apply layerwise relevance propgation 
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

#specific in the function name stands for "specific nodes", an allustion to WSR2
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


with open(outpath + model_attack+'_'+ metric+'_class'+'_'+str(0)+'.pkl', 'rb') as file:
      benign_res_0=pickle.load(file)
      benign_res_act_0=pickle.load(file)
      benign_res_x_0=pickle.load( file)
      adv_res_0=pickle.load( file)
      adv_res_act_0=pickle.load( file)
      adv_res_x=pickle.load( file)
      benign_res_0=np.asarray(benign_res_0)
      adv_res_0=np.asarray(adv_res_0)

      benign_res_act_0 =  [ [el.flatten() for el in  blah] for blah in benign_res_act_0 ]
      benign_res_act_0=[ functools.reduce(lambda a, b: tf.concat([a,b],0),blah) for blah in   benign_res_act_0]
      benign_res_act_0=np.asarray(benign_res_act_0)
      adv_res_act_0 =  [ [el.flatten() for el in  blah] for blah in adv_res_act_0 ]
      adv_res_act_0=[ functools.reduce(lambda a, b: tf.concat([a,b],0),blah) for blah in   adv_res_act_0]
      adv_res_act_0=np.asarray(adv_res_act_0)

      if adv_res_0.shape[0]==0:
          adv_res_0=np.ones(shape=(1,5106))*-10000

with open(outpath+ model_attack+'_'+ metric+'_class'+'_'+str(1)+'.pkl', 'rb') as file:
      benign_res_1=pickle.load(file)
      benign_res_act_1=pickle.load(file)
      benign_res_x_1=pickle.load( file)
      adv_res_1=pickle.load( file)
      adv_res_act_1=pickle.load( file)
      adv_res_x=pickle.load( file)
      benign_res_1=np.asarray(benign_res_1)
      adv_res_1=np.asarray(adv_res_1)

      benign_res_act_1 =  [ [el.flatten() for el in  blah] for blah in benign_res_act_1 ]
      benign_res_act_1=[ functools.reduce(lambda a, b: tf.concat([a,b],0),blah) for blah in   benign_res_act_1]
      benign_res_act_1=np.asarray(benign_res_act_1)
      adv_res_act_1 =  [ [el.flatten() for el in  blah] for blah in adv_res_act_1 ]
      adv_res_act_1=[ functools.reduce(lambda a, b: tf.concat([a,b],0),blah) for blah in   adv_res_act_1]
      adv_res_act_1=np.asarray(adv_res_act_1)

      if adv_res_1.shape[0]==0:
          adv_res_1=np.ones(shape=(1,5106))*-10000

with open(outpath+ model_attack+'_'+ metric+'_class'+'_'+str(2)+'.pkl', 'rb') as file:
      benign_res_2=pickle.load(file)
      benign_res_act_2=pickle.load(file)
      benign_res_x_2=pickle.load( file)
      adv_res_2=pickle.load( file)
      adv_res_act_2=pickle.load( file)
      adv_res_x=pickle.load( file)
      benign_res_2=np.asarray(benign_res_2)
      adv_res_2=np.asarray(adv_res_2)
      
      benign_res_act_2 =  [ [el.flatten() for el in  blah] for blah in benign_res_act_2 ]
      benign_res_act_2=[ functools.reduce(lambda a, b: tf.concat([a,b],0),blah) for blah in   benign_res_act_2]
      benign_res_act_2=np.asarray(benign_res_act_2)
      adv_res_act_2 =  [ [el.flatten() for el in  blah] for blah in adv_res_act_2 ]
      adv_res_act_2=[ functools.reduce(lambda a, b: tf.concat([a,b],0),blah) for blah in   adv_res_act_2]
      adv_res_act_2=np.asarray(adv_res_act_2)

      if adv_res_2.shape[0]==0:
          adv_res_2=np.ones(shape=(1,5106))*-10000

with open(outpath+ model_attack+'_'+ metric+'_class'+'_'+str(3)+'.pkl', 'rb') as file:
      benign_res_3=pickle.load(file)
      benign_res_act_3=pickle.load(file)
      benign_res_x_3=pickle.load( file)
      adv_res_3=pickle.load( file)
      adv_res_act_3=pickle.load( file)
      adv_res_x=pickle.load( file)
      benign_res_3=np.asarray(benign_res_3)
      adv_res_3=np.asarray(adv_res_3)

      benign_res_act_3 =  [ [el.flatten() for el in  blah] for blah in benign_res_act_3 ]
      benign_res_act_3=[ functools.reduce(lambda a, b: tf.concat([a,b],0),blah) for blah in   benign_res_act_3]
      benign_res_act_3=np.asarray(benign_res_act_3)
      adv_res_act_3 =  [ [el.flatten() for el in  blah] for blah in adv_res_act_3 ]
      adv_res_act_3=[ functools.reduce(lambda a, b: tf.concat([a,b],0),blah) for blah in   adv_res_act_3]
      adv_res_act_3=np.asarray(adv_res_act_3)

      if adv_res_3.shape[0]==0:
          adv_res_3=np.ones(shape=(1,5106))*-10000

with open(outpath+ model_attack+'_'+ metric+'_class'+'_'+str(4)+'.pkl', 'rb') as file:
      benign_res_4=pickle.load(file)
      benign_res_act_4=pickle.load(file)
      benign_res_x_4=pickle.load( file)
      adv_res_4=pickle.load( file)
      adv_res_act_4=pickle.load( file)
      adv_res_x=pickle.load( file)
      benign_res_4=np.asarray(benign_res_4)
      adv_res_4=np.asarray(adv_res_4)

      benign_res_act_4 =  [ [el.flatten() for el in  blah] for blah in benign_res_act_4 ]
      benign_res_act_4=[ functools.reduce(lambda a, b: tf.concat([a,b],0),blah) for blah in   benign_res_act_4]
      benign_res_act_4=np.asarray(benign_res_act_4)
      adv_res_act_4 =  [ [el.flatten() for el in  blah] for blah in adv_res_act_4 ]
      adv_res_act_4=[ functools.reduce(lambda a, b: tf.concat([a,b],0),blah) for blah in   adv_res_act_4]
      adv_res_act_4=np.asarray(adv_res_act_4)

      if adv_res_4.shape[0]==0:
          adv_res_4=np.ones(shape=(1,5106))*-10000


with open(outpath+ model_attack+'_'+ metric+'_class'+'_'+str(5)+'.pkl', 'rb') as file:
      benign_res_5=pickle.load(file)
      benign_res_act_5=pickle.load(file)
      benign_res_x_5=pickle.load( file)
      adv_res_5=pickle.load( file)
      adv_res_act_5=pickle.load( file)
      adv_res_act_x=pickle.load( file)
      benign_res_5=np.asarray(benign_res_5)
      adv_res_5=np.asarray(adv_res_5)

      benign_res_act_5 =  [ [el.flatten() for el in  blah] for blah in benign_res_act_5 ]
      benign_res_act_5=[ functools.reduce(lambda a, b: tf.concat([a,b],0),blah) for blah in   benign_res_act_5]
      benign_res_act_5=np.asarray(benign_res_act_5)
      adv_res_act_5 =  [ [el.flatten() for el in  blah] for blah in adv_res_act_5 ]
      adv_res_act_5=[ functools.reduce(lambda a, b: tf.concat([a,b],0),blah) for blah in   adv_res_act_5]
      adv_res_act_5=np.asarray(adv_res_act_5)

      if adv_res_5.shape[0]==0:
          adv_res_5=np.ones(shape=(1,5106))*-10000


with open(outpath+ model_attack+'_'+ metric+'_class'+'_'+str(6)+'.pkl', 'rb') as file:
      benign_res_6=pickle.load(file)
      benign_res_act_6=pickle.load(file)
      benign_res_x_6=pickle.load( file)
      adv_res_6=pickle.load( file)
      adv_res_act_6=pickle.load( file)
      adv_res_x=pickle.load( file)
      benign_res_6=np.asarray(benign_res_6)
      adv_res_6=np.asarray(adv_res_6)

      benign_res_act_6 =  [ [el.flatten() for el in  blah] for blah in benign_res_act_6 ]
      benign_res_act_6=[ functools.reduce(lambda a, b: tf.concat([a,b],0),blah) for blah in   benign_res_act_6]
      benign_res_act_6=np.asarray(benign_res_act_6)
      adv_res_act_6 =  [ [el.flatten() for el in  blah] for blah in adv_res_act_6 ]
      adv_res_act_6=[ functools.reduce(lambda a, b: tf.concat([a,b],0),blah) for blah in   adv_res_act_6]
      adv_res_act_6=np.asarray(adv_res_act_6)

      if adv_res_6.shape[0]==0:
          adv_res_6=np.ones(shape=(1,5106))*-10000

with open(outpath+ model_attack+'_'+ metric+'_class'+'_'+str(7)+'.pkl', 'rb') as file:
      benign_res_7=pickle.load(file)
      benign_res_act_7=pickle.load(file)
      benign_res_x_7=pickle.load( file)
      adv_res_7=pickle.load( file)
      adv_res_act_7=pickle.load( file)
      adv_res_x=pickle.load( file)
      benign_res_7=np.asarray(benign_res_7)
      adv_res_7=np.asarray(adv_res_7)

      benign_res_act_7 =  [ [el.flatten() for el in  blah] for blah in benign_res_act_7 ]
      benign_res_act_7=[ functools.reduce(lambda a, b: tf.concat([a,b],0),blah) for blah in   benign_res_act_7]
      benign_res_act_7=np.asarray(benign_res_act_7)
      adv_res_act_7 =  [ [el.flatten() for el in  blah] for blah in adv_res_act_7 ]
      adv_res_act_7=[ functools.reduce(lambda a, b: tf.concat([a,b],0),blah) for blah in   adv_res_act_7]
      adv_res_act_7=np.asarray(adv_res_act_7)

      if adv_res_7.shape[0]==0:
          adv_res_7=np.ones(shape=(1,5106))*-10000

with open(outpath+ model_attack+'_'+ metric+'_class'+'_'+str(8)+'.pkl', 'rb') as file:
      benign_res_8=pickle.load(file)
      benign_res_act_8=pickle.load(file)
      benign_res_x_8=pickle.load( file)
      adv_res_8=pickle.load( file)
      adv_res_act_8=pickle.load( file)
      adv_res_x=pickle.load( file)
      benign_res_8=np.asarray(benign_res_8)
      adv_res_8=np.asarray(adv_res_8)

      benign_res_act_8 =  [ [el.flatten() for el in  blah] for blah in benign_res_act_8 ]
      benign_res_act_8=[ functools.reduce(lambda a, b: tf.concat([a,b],0),blah) for blah in   benign_res_act_8]
      benign_res_act_8=np.asarray(benign_res_act_8)
      adv_res_act_8 =  [ [el.flatten() for el in  blah] for blah in adv_res_act_8 ]
      adv_res_act_8=[ functools.reduce(lambda a, b: tf.concat([a,b],0),blah) for blah in   adv_res_act_8]
      adv_res_act_8=np.asarray(adv_res_act_8)

      if adv_res_8.shape[0]==0:
          adv_res_8=np.ones(shape=(1,5106))*-10000


with open(outpath+ model_attack+'_'+ metric+'_class'+'_'+str(9)+'.pkl', 'rb') as file:
      benign_res_9=pickle.load(file)
      benign_res_act_9=pickle.load(file)
      benign_res_x_9=pickle.load( file)
      adv_res_9=pickle.load( file)
      adv_res_act_9=pickle.load( file)
      adv_res_x=pickle.load( file)
      benign_res_9=np.asarray(benign_res_9)
      adv_res_9=np.asarray(adv_res_9)

      benign_res_act_9 =  [ [el.flatten() for el in  blah] for blah in benign_res_act_9 ]
      benign_res_act_9=[ functools.reduce(lambda a, b: tf.concat([a,b],0),blah) for blah in   benign_res_act_9]
      benign_res_act_9=np.asarray(benign_res_act_9)
      adv_res_act_9 =  [ [el.flatten() for el in  blah] for blah in adv_res_act_9 ]
      adv_res_act_9=[ functools.reduce(lambda a, b: tf.concat([a,b],0),blah) for blah in   adv_res_act_9]
      adv_res_act_9=np.asarray(adv_res_act_9)

      if adv_res_9.shape[0]==0:
          adv_res_9=np.ones(shape=(1,5106))*-10000





varlist_0, varlist2_0=[], []
for i in range(benign_res_0.shape[1] ):
        varlist_0.append( np.mean(benign_res_0[:,i])  )
        varlist2_0.append( np.mean(adv_res_0[:,i])  )
varlistdiff_0=[-varlist_0[i]+ varlist2_0[i]  for i in range(len(varlist_0)) ]


varlist_1, varlist2_1=[], []
for i in range(benign_res_1.shape[1] ):
        varlist_1.append( np.mean(benign_res_1[:,i])  )
        varlist2_1.append( np.mean(adv_res_1[:,i])  )
varlistdiff_1=[-varlist_1[i]+ varlist2_1[i]  for i in range(len(varlist_1)) ]


varlist_2, varlist2_2=[], []
for i in range(benign_res_2.shape[1] ):
        varlist_2.append( np.mean(benign_res_2[:,i])  )
        varlist2_2.append( np.mean(adv_res_2[:,i])  )
varlistdiff_2=[-varlist_2[i]+ varlist2_2[i]  for i in range(len(varlist_2)) ]


varlist_3, varlist2_3=[], []
for i in range(benign_res_3.shape[1] ):
        varlist_3.append( np.mean(benign_res_3[:,i])  )
        varlist2_3.append( np.mean(adv_res_3[:,i])  )
varlistdiff_3=[-varlist_3[i]+ varlist2_3[i]  for i in range(len(varlist_3)) ]



varlist_4, varlist2_4=[], []
for i in range(benign_res_4.shape[1] ):
        varlist_4.append( np.mean(benign_res_4[:,i])  )
        varlist2_4.append( np.mean(adv_res_4[:,i])  )
varlistdiff_4=[-varlist_4[i]+ varlist2_4[i]  for i in range(len(varlist_4)) ]



varlist_5, varlist2_5=[], []
for i in range(benign_res_5.shape[1] ):
        varlist_5.append( np.mean(benign_res_5[:,i])  )
        varlist2_5.append( np.mean(adv_res_5[:,i])  )
varlistdiff_5=[-varlist_5[i]+ varlist2_5[i]  for i in range(len(varlist_5)) ]


varlist_6, varlist2_6=[], []
for i in range(benign_res_6.shape[1] ):
        varlist_6.append( np.mean(benign_res_6[:,i])  )
        varlist2_6.append( np.mean(adv_res_6[:,i])  )
varlistdiff_6=[-varlist_6[i]+ varlist2_6[i]  for i in range(len(varlist_6)) ]


varlist_7, varlist2_7=[], []
for i in range(benign_res_7.shape[1] ):
        varlist_7.append( np.mean(benign_res_7[:,i])  )
        varlist2_7.append( np.mean(adv_res_7[:,i])  )
varlistdiff_7=[-varlist_7[i]+ varlist2_7[i]  for i in range(len(varlist_7)) ]


varlist_8, varlist2_8=[], []
for i in range(benign_res_8.shape[1] ):
        varlist_8.append( np.mean(benign_res_8[:,i])  )
        varlist2_8.append( np.mean(adv_res_8[:,i])  )
varlistdiff_8=[-varlist_8[i]+ varlist2_8[i]  for i in range(len(varlist_8)) ]


varlist_9, varlist2_9=[], []
for i in range(benign_res_9.shape[1] ):
        varlist_9.append( np.mean(benign_res_9[:,i])  )
        varlist2_9.append( np.mean(adv_res_9[:,i])  )
varlistdiff_9=[-varlist_9[i]+ varlist2_9[i]  for i in range(len(varlist_9)) ]



#construct the set S
var_inds=[]
for i in range(benign_res_9.shape[1] ):
        if varlistdiff_0[i ] <0 and varlistdiff_1[i ] <0 and varlistdiff_2[i ] <0 and varlistdiff_3[i ] <0 and varlistdiff_4[i ] <0 and varlistdiff_5[i ] <0 and varlistdiff_6[i ] <0 and varlistdiff_7[i ] <0 and varlistdiff_8[i ] <0 and varlistdiff_9[i ] <0:
                var_inds.append(i)



#######compute WSR1 or WSR2###############################################

activations=False #Do not change, this was for a different experiment.
WSR2=True
if WSR2==True:
    test_benign=False
    dist1_adv,dist2_adv=test_classification_specific(range(7000,10000), activations)
    test_benign=True
    dist1_benign,dist2_benign=test_classification_specific(range(7000,1000),activations)
    with open(outpath+ model_attack  + str(activations) + 'WSR2.pkl'  , 'wb') as file:
        pickle.dump(dist1_adv, file)
        pickle.dump(dist2_adv, file)
        pickle.dump( dist1_benign, file)
        pickle.dump(dist2_benign, file)
else:
    test_benign=False
    dist1_adv,dist2_adv=test_classification(range(7000,10000), activations)
    test_benign=True
    dist1_benign,dist2_benign=test_classification(range(7000,10000),activations)
    with open(outpath+ model_attack   + str(activations) + 'WSR1.pkl', 'wb') as file:
        pickle.dump(dist1_adv, file)
        pickle.dump(dist2_adv, file)
        pickle.dump( dist1_benign, file)
        pickle.dump(dist2_benign, file)


#plot WSR
fig, axs = plt.subplots(nrows=2,ncols=1,sharex=True)  
axs[0].hist([dist1_adv[i]/dist2_adv[i] for i in range(len(dist1_adv))],50)
axs[1].hist([dist1_benign[i]/dist2_benign[i] for i in range(len(dist1_benign))],50)
axs[0].set_title('WSR: adversarial samples')
axs[1].set_title('WSR: benign samples')



###### end compute WSR1 or WSR2###########################################



##############Compute area under receiver operating characteristic (AUROC) curve. The two curves are the histograms of val1 and val2.###########################################
val1=[dist1_adv[i]/dist2_adv[i] for i in range(len(dist1_adv))]
val2=[dist1_benign[i]/dist2_benign[i] for i in range(len(dist1_benign))]
fp=[]
tp=[]
for num in np.linspace(0,5,1000):
    val1_count=len([el for el in val1 if el<num])
    val2_count=len([el for el  in val2 if el<num])
    
    fp_prob=val1_count/len(val1)
    tp_prob=val2_count/len(val2)
    fp.append(fp_prob)
    tp.append(tp_prob)

fp=np.asarray(fp)
tp=np.asarray(tp)
one_inds=np.where(tp==1)
first_one=one_inds[0][0]
fp=fp[0:first_one+1]
tp=tp[0:first_one+1]

zero_inds=np.where(tp==0)
last_zero=zero_inds[0][-1]
fp=fp[last_zero:]
tp=tp[last_zero:]

1-metrics.auc(tp, fp)
