# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 10:31:32 2021

@author: mitran
"""

import torch
import numpy as np



class Anchors_fpn():
  def __init__(self,image_size=256,fpn_level=[3,4,5,6,7]):
       
       self.size=image_size
       self.ims=image_size
       self.fpn=fpn_level
       self.stride=[2**x for x in self.fpn]
       self.no_pixels=[int(np.ceil(image_size/x)) for x in self.stride]
  def get_centers(self,val,stride):
       mat=list()
       for i in range(val):
         for j in range(val):
            mat.append((int((i+0.5)*stride),int((j+0.5)*stride)))
       return mat
  def __call__(self,device):
       Anchors=[]
       for i,s in enumerate(self.stride):
           l=self.fpn[i]
           size=[s*2**(l-3),s*2**(l-2.5),s*2**(l-2)] # check after wards
           ratio=[2,3]
           print(size,s,l)
           #print("Anchors for",s,len(self.get_centers(self.no_pixels[i],s)))
           for center in self.get_centers(self.no_pixels[i],s):

             for sze in size:
               anc=[center[0],center[1],sze,sze]
               Anchors.append(anc)
               for rat in ratio:
                    anc=[center[0],center[1],sze*rat,sze*(1/rat)**0.5]
                    Anchors.append(anc)
                    anc=[center[0],center[1],sze*(1/rat)**0.5,sze*rat]
                    Anchors.append(anc)

       Anchors=torch.tensor((Anchors),dtype=torch.float).to(device)/self.ims
       return Anchors


def jaccard(box_a, box_b):
   
    A = box_a.size(0)
    B = box_b.size(0)
    
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    
    inter = torch.clamp((max_xy - min_xy), min=0)

    inter= inter[:, :, 0] * inter[:, :, 1]

    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def encode_bbox(matched, priors):
    
    

    x=(matched[:, :2] - priors[:, :2]) / priors[:, 2:]
    y=torch.log((matched[:, 2:] / priors[:, 2:]))

    return torch.cat([x,y], 1) 

def box_Decode(prediction,anchor):
      anchor=anchor.cpu().detach().numpy()
      anc_h = anchor[:, 3] 
      anc_w = anchor[:, 2] 
      anc_y = anchor[:, 1] 
      anc_x = anchor[:, 0]

      pred=prediction.cpu().detach().numpy()
      dx = pred[:, 0] 
      dy = pred[:,1 ] 
      dh = pred[:,3] 
      dw = pred[:,2] 

      y = dy * anc_h + anc_y
      x = dx * anc_w + anc_x
      h = np.exp(dh) * anc_h
      w = np.exp(dw) * anc_w


     
      return np.stack([x,y,w,h],axis=1)
  
    
def cvt_chw(box):
  height = box[:, 3] - box[:, 1]
  width = box[:, 2] - box[:, 0]
  y = box[:, 1] + 0.5 * height
  x = box[:, 0] + 0.5 * width
  return torch.stack([x,y,width,height],dim=1)

def match_anchors(self,class_t,bbox_t,kptns_t,thresh):
  
  # find iou for all anchors with all boxes
  overlap_iou_matrix=jaccard(bbox_t,self.generated_anchors_minmax)# target bbox is created by min max and jaccard calculates using that
  #print(bbox_t.shape,self.generated_anchors.shape,overlap_iou_matrix.shape)
  #torch.Size([1, 4]) torch.Size([26610, 4]) torch.Size([1, 26610])
  #boundingbox,no_Anchors

  maxi,_=overlap_iou_matrix.max(1)
  #print((maxi>0.2).long().sum())
  if((maxi>thresh).long().sum()==0):
    #print("no bounding box for this batch")
    #print(maxi,bbox_t)
    #assert(1==2)
    return []


  #assign the bounding box to the anchor having high coverage 

  #  iou val for which the anchor merges the most with the available bounding box    , the id of the bouding box to whcih the anchor merged the mose
  max_iou_anchor,max_iou_anchorid =overlap_iou_matrix.max(0) 

  #print(max_iou_anchor.shape,max_iou_anchorid.shape)

  

  #the bounding box coordinates to which the anchor had more overlap
  bbox_t=cvt_chw(bbox_t)

  bbox_anchors_more_iou=bbox_t[max_iou_anchorid]
  inter_bbox=encode_bbox(bbox_anchors_more_iou,self.generated_anchors)

 

  # retrive the class from the id of the bounding box having the more overlap(background has 0 overlap with first box asn it is taken for the class of the anchors) and they are remove with the help of iou value and thresh 
  class_anchors_more_iou=class_t[max_iou_anchorid]
  #background class which implies there is no object in the box
  class_anchors_more_iou[max_iou_anchor<thresh]=0

  #print(inter_bbox.shape,inter_landm.shape,class_anchors_more_iou.shape)
  return [class_anchors_more_iou,inter_bbox]