# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 10:01:37 2021

@author: mitran
"""


import cv2
import torch
from torch.nn import *
import os


from torch.nn import functional as F








class Anchor_bbox(Module):
  def __init__(self,dimension,num_anchors):
    super().__init__()
    self.conv=Conv2d(dimension,num_anchors*4,1)
  def forward(self,feature_map):
    y=self.conv(feature_map)
    batch_sze=y.shape[0]
    y=y.permute(0,2,3,1).contiguous().view(batch_sze,-1,4)
    return y


class Anchor_class(Module):
  def __init__(self,dimension,num_anchors):
    super().__init__()
    self.conv=Conv2d(dimension,num_anchors*2,1)
  def forward(self,feature_map):
    y=self.conv(feature_map)
    batch_sze=y.shape[0]
    y=y.permute(0,2,3,1).contiguous().view(batch_sze,-1,2)
    return y











def chw2minmax(center_box):

  assert (center_box.shape[-1]==4)
  cx=center_box[:,0]
  cy=center_box[:,1]
  width=torch.div(center_box[:,2],2)
  height=torch.div(center_box[:,3],2)

  minx=cx-width
  miny=cy-height
  maxx=cx+width
  maxy=cy+height
  boxes=torch.stack([minx,miny,maxx,maxy],dim=1)
  return boxes

class SSD(Module):
  def __init__(self,backbone,fpn_diminsion=64,no_landmarks=4,thres=0.5,device='cpu'):
    super(SSD,self).__init__()
    self.backbone=backbone
    self.fpn_levels=3
    self.FPN=Feature_Pyramid_Network([64,128,256],fpn_diminsion,self.fpn_levels)
    self.no_anchors=15
    self.device=torch.device(device)
    self.anchor_class=ModuleList([Anchor_class(fpn_diminsion,self.no_anchors) for i in range(self.fpn_levels)])
    self.anchor_bbbox=ModuleList([Anchor_bbox(fpn_diminsion,self.no_anchors) for i in range(self.fpn_levels)])
    #self.ssh=SSH(fpn_diminsion,self.fpn_levels)
    anchors=Anchors_fpn(image_size=256,fpn_level=[i for i in range(3,3+self.fpn_levels)])
    x=anchors(self.device)  
    self.generated_anchors=x
    self.generated_anchors_minmax=chw2minmax(x)
    self.lm=no_landmarks
    self.no_neg_ratio=3
    self.thresh=thres
    

 
  def forward(self,input):
    #print(input.shape)
    feature_maps=self.backbone(input)
    #print([i.shape for i in feature_maps])
    feature_pyramids=self.FPN(feature_maps)
    #feature_pyramids=self.ssh(feature_pyramids)
    

    classifications = torch.cat([self.anchor_class[i](feature) for i, feature in enumerate(feature_pyramids)],dim=1)
    bbox_regressions = torch.cat([self.anchor_bbbox[i](feature) for i, feature in enumerate(feature_pyramids)], dim=1)    

    return[classifications,bbox_regressions]

  def accuracy(self,predictions, labels):
      classes = torch.argmax(predictions, dim=1)
      return torch.mean((classes == labels).float())     
                                       

RetinaFace.match_anchors=match_anchors
RetinaFace.Multibox_loss=Multibox_loss


   









