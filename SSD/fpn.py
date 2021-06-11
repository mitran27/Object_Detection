# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 10:34:39 2021

@author: mitran
"""

import torch
from torch.nn import *


class Feature_Pyramid_Network(Module):
  def __init__(self,backbone_filter,feature_size=256,fpnlvl=5):
    super().__init__()
    self.fpnlvl=fpnlvl

    self.pyr3=conv_1x1(backbone_filter[0],feature_size)
    self.pyr3_m=conv_3x3(feature_size)
    

    self.pyr4=conv_1x1(backbone_filter[1],feature_size)
    self.pyr4_m=conv_3x3(feature_size)
    self.pyr4_up=Upsample(scale_factor=2, mode='nearest')

    self.pyr5=conv_1x1(backbone_filter[2],feature_size)
    self.pyr5_m=conv_3x3(feature_size)
    self.pyr5_up=Upsample(scale_factor=2, mode='nearest')
    if(fpnlvl==5):
        self.pyr6=Conv2d(backbone_filter[2],feature_size,3,2,padding=1)
        slope=0 if feature_size==256 else 0.1
        self.act=LeakyReLU(slope)
        self.pyr7=Conv2d(feature_size,feature_size,3,2,padding=1)
  def forward(self,backbone_features):


    pyramid_5= self.pyr5(backbone_features[2])
    temp=self.pyr5_up(pyramid_5)
    
    pyramid_4=self.pyr4(backbone_features[1])
    pyramid_4=temp+pyramid_4
    temp=self.pyr4_up(pyramid_4)
    
    pyramid_3=self.pyr3(backbone_features[0])
    pyramid_3=temp+pyramid_3

    pyramid_3=self.pyr3_m(pyramid_3)
    pyramid_4=self.pyr4_m(pyramid_4)
    pyramid_5=self.pyr5_m(pyramid_5)

    if(self.fpnlvl==3):
        return  [pyramid_3,pyramid_4,pyramid_5]   

    pyramid_6=self.pyr6(backbone_features[2])
    
    x=self.act(pyramid_6)
    pyramid_7=self.pyr7(x)
    
    return [pyramid_3,pyramid_4,pyramid_5,pyramid_6,pyramid_7]
     
    