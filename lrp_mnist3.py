# -*- coding: utf-8 -*-
#Adapted from https://github.com/atulshanbhag/Layerwise-Relevance-Propagation
#Instead of only returning relevances for activations, the functions here also return relevances along weights between neurons.
import sys

from tensorflow.keras                        import backend as K
from tensorflow.python.ops        import gen_nn_ops
from utils_lrp                    import (get_model_params, 
                                          get_gammas, 
                                          get_heatmaps, 
                                          load_images,
                                          predict_labels, 
                                          visualize_heatmap)                   

import numpy as np
import matplotlib.pyplot as plt


import tensorflow as tf



class LayerwiseRelevancePropagation:

  def __init__(self, model, alpha=2, epsilon=1e-7, lrp_sign='both', lrp_formula=3 ):
    #model_name = model_name.lower()
    #if model_name == 'vgg16':
    #  model_type = VGG16
    #elif model_name == 'vgg19':
    #  model_type = VGG19
    #else:
    #  raise 'Model name not one of VGG16 or VGG19'
    #  sys.exit()
    #self.model = model_type(include_top=True, weights='imagenet', input_shape=(224, 224, 3))
    self.model=model
    self.alpha = alpha
    self.beta = 1 - alpha
    self.epsilon =epsilon
    self.lrp_sign=lrp_sign
    self.lrp_formula=lrp_formula
    


    self.names, self.activations, self.weights = get_model_params(self.model)
    self.num_layers = len(self.names)

    self.relevance, self.relevance_weights = self.compute_relevances()
    self.lrp_runner= K.function(inputs=[self.model.input], outputs=[self.relevance, self.relevance_weights])
    self.lrp_runner_activations= K.function(inputs=[self.model.input, ], outputs=self.activations)
    
  

  def compute_relevances(self):
    r = self.model.output
    relevance_activs=[]
    rel_weights=[]
    
    for i in range(self.num_layers-2, -2, -1):
      if i !=-1:
          print(i)
          if 'fc' in self.names[i + 1]:
            print('fc')
            
            r, rw= self.backprop_fc(self.weights[i + 1][0], self.weights[i + 1][1], self.activations[i],self.activations[i+1] , r)
         
            relevance_activs.append(r)
            rel_weights.append(rw)
          elif 'flatten' in self.names[i + 1]:
            print('flatten')
            r,rw= self.backprop_flatten(self.activations[i], r, rel_weights[-1])
            
            relevance_activs.append(r)
            rel_weights.append(rw)
         
          elif 'pool' in self.names[i + 1]:
            print('pool')
            r,rw = self.backprop_max_pool2d(self.activations[i],self.activations[i+1], r)
            relevance_activs.append(r)
            rel_weights.append(rw)
          elif 'conv' in self.names[i + 1]:
            print('conv')
            
            r,rw = self.backprop_conv2d(self.weights[i + 1][0], self.weights[i + 1][1], self.activations[i],self.activations[i+1], r)
           
            relevance_activs.append(r)
            rel_weights.append(rw)
            
          else:
            raise 'Layer not recognized!'
            sys.exit()
      else:
          if 'conv' in self.names[i + 1]:
            print('conv')
            r,rw = self.backprop_conv2d(self.weights[i + 1][0], self.weights[i + 1][1], self.model.input,self.activations[i+1], r)
            relevance_activs.append(r)
            rel_weights.append(rw)
         
    
    
    return relevance_activs, rel_weights



  def backprop_fc(self, w, b, a, a2, r):
    if self.lrp_sign=='positive':
        w = K.maximum(w, 0.)
        b = K.maximum(b, 0.)
        z = K.dot(a, w) + b + self.epsilon
        s = r / z
        s=tf.convert_to_tensor(s)
        c = K.dot(s, K.transpose(w))

        s_prime=tf.expand_dims(s, axis=1)
        rw=tf.math.multiply( w, s_prime)
        a_prime=tf.expand_dims(a, axis=2)
        
        if self.lrp_formula== 1:
            return a *  c , a_prime*rw
        elif self.lrp_formula== 2:
            return a *  c , rw    
        elif self.lrp_formula== 3:
            return a *  c , rw/a2
        

    if self.lrp_sign=='negative':
       # pdb.set_trace()
        w = K.minimum(w, 0.)
        b = K.minimum(b, 0.)
        z = K.dot(a, w) + b - self.epsilon
        s = r / z
        s=tf.convert_to_tensor(s)
        c = K.dot(s, K.transpose(w))
        
        s_prime=tf.expand_dims(s, axis=1)
        rw=tf.math.multiply( w, s_prime)
        a_prime=tf.expand_dims(a, axis=2)
        
        if self.lrp_formula== 1:
            return a *  c , a_prime*rw
        elif self.lrp_formula== 2:
            return a *  c , rw    
        elif self.lrp_formula== 3:
            return a *  c , rw/a2
 
    if self.lrp_sign=='both':
        w_p = K.maximum(w, 0.)
        b_p = K.maximum(b, 0.)
        z_p = K.dot(a, w_p) + b_p + self.epsilon
        s_p = r / z_p
        s_p=tf.convert_to_tensor(s_p)
        c_p = K.dot(s_p, K.transpose(w_p))

        w_n = K.minimum(w, 0.)
        b_n = K.minimum(b, 0.)
        z_n = K.dot(a, w_n) + b_n - self.epsilon
        s_n = r / z_n
        s_n=tf.convert_to_tensor(s_n)
        c_n = K.dot(s_n, K.transpose(w_n))
        
        s_prime_p=tf.expand_dims(s_p, axis=1)
        rw_p=tf.math.multiply( w_p, s_prime_p)
        s_prime_n=tf.expand_dims(s_n, axis=1)
        rw_n=tf.math.multiply( w_n, s_prime_n)
        
        a_prime=tf.expand_dims(a, axis=2)
        
        if self.lrp_formula== 1:
            return  a * (self.alpha * c_p + self.beta * c_n) , a_prime*(rw_p+rw_n)
        elif self.lrp_formula== 2:
            return a * (self.alpha * c_p + self.beta * c_n) , (rw_p+rw_n)   
        elif self.lrp_formula== 3:
            return a * (self.alpha * c_p + self.beta * c_n) , a_prime*(rw_p+rw_n)/a2
        
    if self.lrp_sign=='no_sign':
        w = K.minimum(w, 100000000.)
        b = K.minimum(b, 100000000.)
        z = K.dot(a, w) + b - self.epsilon
        s = r / z
        c = K.dot(s, K.transpose(w))
        s_prime=tf.expand_dims(s, axis=1)
    
        rw=tf.math.multiply( w, s_prime)
        a_prime=tf.expand_dims(a, axis=2)
        
        if self.lrp_formula== 1:
            return a *  c , a_prime*rw
        elif self.lrp_formula== 2:
            return a *  c , rw    
        elif self.lrp_formula== 3:
            return a *  c , rw/a2
   


  def backprop_flatten(self, a, r, rw):
      
    shape = a.get_shape().as_list()
    
    shape[0] = -1
    #print('shape after', shape)
    shape2 = rw.get_shape().as_list()
    blah=shape2[0]
    #print('shape ', shape, 'shape2 ', shape2)
    newshape=shape+[shape2[-1]]
    #print('newshape ', newshape)
    #print('rwshape ', rw.shape)
    #print('rwnewshape', K.reshape(rw , newshape))
    #return K.reshape(r, shape),
    

    return K.reshape(r, shape), K.reshape(r, shape)
    

  def backprop_max_pool2d(self, a, a2, r, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1)):
    z = K.pool2d(a, pool_size=ksize[1:-1], strides=strides[1:-1], padding='VALID', pool_mode='max')

    z_p = K.maximum(z, 0.) + self.epsilon

    s_p = r / z_p
    c_p = gen_nn_ops.max_pool_grad_v2(a, z_p, s_p, ksize, strides, padding=padding)

    z_n = K.minimum(z, 0.) - self.epsilon
    s_n = r / z_n
    c_n = gen_nn_ops.max_pool_grad_v2(a, z_n, s_n, ksize, strides, padding=padding)
    
    #dwight modified the formula
    z_tmp = K.minimum(z, 1000000.) + self.epsilon
    s = r / z_tmp
    c = gen_nn_ops.max_pool_grad_v2(a, z_tmp, s, ksize, strides, padding=padding)
    
    if self.lrp_formula== 1:
        return a *  c , tf.tensordot((a*c),s,axes=0)
    elif self.lrp_formula== 2:
        return a *  c , tf.tensordot(c,s,axes=0)   
    elif self.lrp_formula== 3:
        return a *  c , tf.tensordot(c,s/a2,axes=0)
    
    #return a * (self.alpha * c_p + self.beta * c_n)



  def backprop_conv2d(self, w, b, a,a2, r, strides=(1, 1, 1, 1)):
  

    w_p = K.maximum(w, 0.)
    b_p = K.maximum(b, 0.)
    z_p = K.conv2d(a, kernel=w_p, strides=strides[1:-1], padding='SAME') + b_p + self.epsilon

    #r=tf.reshape(r, [-1]) #dwight
    #z_p=tf.reshape(z_p, [-1]) #dwight

    s_p = r / z_p

    c_p =tf.compat.v1.nn.conv2d_backprop_input(K.shape(a), w_p, s_p, strides, padding=padding)

    w_n = K.minimum(w, 0.)
    b_n = K.minimum(b, 0.)
    z_n = K.conv2d(a, kernel=w_n, strides=strides[1:-1], padding=padding) + b_n - self.epsilon
    #z_n=tf.reshape(z_n, [-1]) #dwight
    s_n = r / z_n
    c_n =tf.compat.v1.nn.conv2d_backprop_input(K.shape(a), w_n, s_n, strides, padding=padding)
    
    #dwight modified the formula
    w_tmp = K.minimum(w, 1000000.)
    b_tmp = K.minimum(b, 1000000.)
    z = K.conv2d(a, kernel=w_tmp, strides=strides[1:-1], padding=padding) + b_tmp - self.epsilon
    s = r / z
    
    c =tf.compat.v1.nn.conv2d_backprop_input(K.shape(a), w_tmp, tf.ones_like(z), strides, padding=padding)
    
    if self.lrp_formula== 1:
        return a *  c , tf.tensordot((a*c),s,axes=0)
    elif self.lrp_formula== 2:
        return a *  c , tf.tensordot(c,s,axes=0)
    elif self.lrp_formula== 3:
        return a *  c , tf.tensordot(c,s/a2,axes=0)
    
    
   # c = gen_nn_ops.max_pool_grad_v2(a, z, s, ksize, strides, padding=padding)

    #s_prime=tf.expand_dims(s, axis=1)
    #rw=tf.math.multiply( w_tmp, s_prime)

    #return a * (self.alpha * c_p + self.beta * c_n)

    #return a * c, rw


  def predict_labels(self, images):
    return predict_labels(self.model, images)

  def run_lrp(self, images):
    print("Running LRP on {0} images...".format(len(images)))
    return self.lrp_runner(images)[0]

  def compute_heatmaps(self, images, g=0.2, cmap_type='rainbow', **kwargs):
    lrps = self.run_lrp(images)
    lrps=lrps[0]
    lrps_shape=lrps.shape
    lrps=np.reshape(lrps, (28,28)   )

    return lrps




if __name__ == '__main__':
 
    
#shape of each element of acivs_rel is (num samples, num_activations)
    lrp = LayerwiseRelevancePropagation(model)
    relevances=lrp.lrp_runner(x_test[0:5])
    activs_rel=relevances[0]
    weights_rel=relevances[1]

    #for cifar data
    # for img, hmap, label in zip(x_test[0:3], heatmaps, y_test[0:3]):
    #     visualize_heatmap(img, hmap, label)fig = plt.figure()
    
    
   # for mnist data
    heatmaps = lrp.compute_heatmaps(x_test[0:5])
    for img, hmap, label in zip(x_test[0:5], heatmaps, y_test[0:5]):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.imshow(hmap, cmap='Reds', interpolation='bilinear')

