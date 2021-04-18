# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 09:43:29 2021

@author: mitran
"""

# Visual Speech Recognition 
   
import tensorflow as tf
from tensorflow.keras.models import Model as Module
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Conv3D,Bidirectional,BatchNormalization,ReLU,MaxPooling3D,concatenate,Input,Activation,AveragePooling3D                             
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import Sequence as Dataloader
import cv2
import dlib
import os

from keras import backend as F



def Convolutional_block(x,filters,kernel,strides=(1, 1, 1),name="none"):
    
    """convolution-> norm -> activation   a standard for CNN"""
    x = Conv3D(filters,kernel,strides=strides,padding="same",use_bias=False,name=name + '_conv')(x)
    x = BatchNormalization(axis=-1, scale=False, name=name + '_bn')(x)
    x = Activation('relu', name=name)(x)

    return x
def Inception_block(x,name,filters):
    """
    Input : sequential image()
    Output: Image after Inception architct Convolutions
    
    Process:
        1) image is passed through four components and concat at the end
    
       component 1)   image --> 1x1 convolution
       component 2)   image --> 1x1 convolution --> 3x3 convolution
       component 3)   image --> 1x1 convolution --> 3x3 convolution 
       component 4)   image --> 3x3 max pool --> 1x1 convolution 
      
       2) concat all 4 components for passing to next layer
    """
    f0a,f1a,f1b,f2a,f2b,f3b=filters
    
    cmp1 = Convolutional_block(x, f0a,( 1, 1, 1), name='Conv3d_'+name+'_0a_1x1')

    cmp2 = Convolutional_block(x, f1a, (1, 1, 1), name='Conv3d_'+name+'_1a_1x1')
    cmp2 = Convolutional_block(cmp2, f1b, (3, 3, 3), name='Conv3d_'+name+'_1b_3x3')

    cmp3 = Convolutional_block(x, f2a, (1, 1, 1), name='Conv3d_'+name+'_2a_1x1')
    cmp3 = Convolutional_block(cmp3, f2b, (3, 3, 3), name='Conv3d_'+name+'_2b_3x3')

    cmp4 = MaxPooling3D((3, 3, 3), (1, 1, 1),padding="same", name='MaxPool2d_'+name+'_3a_3x3')(x)
    cmp4 = Convolutional_block(cmp4, f3b, (1, 1, 1), name='Conv3d_'+name+'_3b_1x1')
    y = concatenate(
        [cmp1, cmp2, cmp3, cmp4],
        axis=-1,
        name='Mixed_'+name)
    
    return y
    
   

def Inception_Inflated3d(inputs,Filtersdict):
   

    """
    Architecture taken from 
    Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset figure 3
    and Filtersdict is from Googlelenet version 1

    """

    x = Convolutional_block(inputs, 64, (7, 7, 7), strides=(2, 2, 2), name='Conv3d_1a_7x7')

    x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), name='MaxPool2d_2a_3x3')(x)
    x = Convolutional_block(x, 64, (1, 1, 1), strides=(1, 1, 1), name='Conv3d_2b_1x1')
    x = Convolutional_block(x, 192, (3, 3, 3), strides=(1, 1, 1), name='Conv3d_2c_3x3')
    x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), name='MaxPool2d_3a_3x3')(x)
    arch=Filtersdict
    
    x=Inception_block(x,'3b',arch['3b'])
    x=Inception_block(x,'3c',arch['3c'])
     
    x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding='same', name='MaxPool2d_4a_3x3')(x)

    x=Inception_block(x,'4b',arch['4b'])


   
    x=Inception_block(x,'4c',arch['4c'])


    x=Inception_block(x,'4d',arch['4d'])

    x=Inception_block(x,'4e',arch['4e'])
    
    
    x=Inception_block(x,'4f',arch['4f'])

    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same', name='MaxPool2d_5a_2x2')(x)

    x=Inception_block(x,'5b',arch['5b'])
    
    x=Inception_block(x,'5c',arch['5c'])

   
    x = AveragePooling3D((2, x.shape[2], x.shape[3]), strides=(1, 1, 1), padding='valid', name='global_avg_pool')(x)
    return Module(inputs, x, name='i3d_inception')


inceptionv1arch={}

inceptionv1arch['order']=['0a','1a','1b','2a','2b','3b']# 3a is max pooling
inceptionv1arch['3b']=   [64,96,128,16,32,32]
inceptionv1arch['3c']=   [128,128,192,32,96,64]
inceptionv1arch['4b']=   [192,96,208,16,48,64]
inceptionv1arch['4c']=   [160,112,224,24,64,64]
inceptionv1arch['4d']=   [128,128,256,24,64,64]
inceptionv1arch['4e']=   [112,144,288,32,64,64]
inceptionv1arch['4f']=   [256,160,320,32,128,128]
inceptionv1arch['5b']=   [256,160,320,32,128,128]
inceptionv1arch['5c']=   [384,192,384,48,128,128]

i=Input(shape=(30,256,256,3),name="input_1")
model=Inception_Inflated3d(i,inceptionv1arch)

model.load_weights("i3dpre.h5")

"""
import h5py
f = h5py.File('i3dpre1.h5', 'r')
a1=list()
for k,v in f.items():
    for k1,v1 in v.items():
        for k2,v2 in v1.items():
            a1.append([k,k1,k2,v2.shape])
f = h5py.File('i3dpre.h5', 'r')
a2=list()
for k,v in f.items():
    for k1,v1 in v.items():
        for k2,v2 in v1.items():
            a2.append([k,k1,k2,v2.shape])

for i,j in zip(a1,a2):
    if(i!=j):
       print(i,j)
       


"""
 