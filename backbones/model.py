# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 15:59:40 2021

@author: mitran
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from torch.nn import CrossEntropyLoss,BCELoss
import math
import numpy as np
import matplotlib.pyplot as plt
import lightly

from torch import nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from concurrent.futures import ThreadPoolExecutor,as_completed
from sklearn.manifold import TSNE
import torch.nn.functional as F
from torch.nn import Module,Linear,Dropout,LayerNorm,BatchNorm2d,conv2d,LeakyReLU,MaxPool2d,GRU
import torchvision.models as models
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor,as_completed
from onnxruntime import InferenceSession, SessionOptions, get_all_providers 
import albumentations as aug
from memory_profiler import profile
from datetime import datetime
import cv2
import soundfile as sf
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from torch.utils.data import DataLoader
from torch.nn import Module,Sequential  
from torch.nn import Dropout,Linear,Flatten,Softmax,Conv2d,LSTM,AdaptiveAvgPool2d
from torch.optim import Adagrad,Adam
import random
import torch
import librosa



    

 

class Identiy(Module):
   def __init__(self):
      super().__init__()
      pass
   def forward(self,X):
        return X   
    
    
class Resnet2D(Module):
   def __init__(self,filters):
      super().__init__()
      self.conv1=conv2d(filters[0],filters[1],kernel_size=(3),padding=1)
      self.norm1=BatchNorm2d(num_features=filters[1])
      self.act1=LeakyReLU()

      self.conv2=conv2d(filters[1],filters[1],kernel_size=(3),padding=1)
      self.norm2=BatchNorm2d(num_features=filters[1])

      if(filters[0]!=filters[1]): 
           self.convr=conv2d(filters[0],filters[1],kernel_size=(1),padding=0)
      else:
          self.convr=Identiy()     
      self.normr=BatchNorm2d(num_features=filters[1])
      self.actr=LeakyReLU()

      self.pool=MaxPool2d(3)



   def forward(self,X):

     y=self.conv1(X)
     y=self.norm1(y)
     y=self.act1(y)

     y=self.conv2(y)
     y=self.norm2(y)

     res=self.convr(X)
     res=self.normr(res)

     y=torch.add(y,res)
     y=self.actr(y)

     y=self.pool(y)  
     

     return y   


class Model(Module):
   def __init__(self,no_clas):
     super().__init__() 

     self.block1=Resnet2D([128,128])
     self.block2=Resnet2D([128,128]) 

     self.block3=Resnet2D([128,256])
     self.block4=Resnet2D([256,256])
     self.block5=Resnet2D([256,256])
     self.block6=Resnet2D([256,256])

     self.block7=Resnet2D([256,512])
     self.block8=Resnet2D([512,512])
     self.block9=Resnet2D([512,512])
     self.block10=Resnet2D([512,512])

     
     
     
   
    
   
  
    
   def forward(self,X):

     y=self.block1(X)
     y=self.block2(y) 

     y=self.block3(y)  
     y=self.block4(y)  
     y=self.block5(y)  
     y=self.block6(y)  

     y=self.block7(y)
     y=self.block8(y)
     y=self.block9(y)  
     y=self.block10(y)


     
     return y 
     

