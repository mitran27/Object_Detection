# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 10:30:06 2021

@author: mitran
"""
import torch
from torch.nn import functional as F

def Multibox_loss(self,prediction,target):    

    class_pred,bbbox_pred,landm_pred=prediction
    class_truth,bbbox_truth,landm_truth=target
  

    batch_sze=class_pred.shape[0]
    assert(class_pred.shape[0]==class_truth.shape[0])


    # finding overlap(iou) iteratively for each batch

    #intermediate -> output of the convolutions and target cannot be compared directly so targets are bought to a itermediate state(removing influence of centers and taking only distribution )

    class_inter=torch.empty(class_pred.shape[:-1],dtype=torch.long).to(self.device)# class pred will not have last axis for cross entropy
    bbbox_inter=torch.empty(bbbox_pred.shape).to(self.device)
    landm_inter=torch.empty(landm_pred.shape).to(self.device)

    for b in range(batch_sze):

        """
        find intermediate data 
        find iou for all anchors with all boxes
        assign the bounding box to the anchor having high coverage
        to remove influence of center encode them (Rcnn paper) 
        the anchors having less coverage than thresh  with all objects present are considered as background
        
        """
             
        inter_datas= self.match_anchors(class_truth[b],bbbox_truth[b],landm_truth[b],self.thresh)
        if(len(inter_datas)==0):
          class_inter[b]=torch.zeros(class_inter.shape[1:])
          bbbox_inter[b]=torch.zeros(bbbox_inter.shape[1:])
          landm_inter[b]=torch.zeros(landm_inter.shape[1:])
          continue

        #print(inter_datas[0].shape,class_inter.shape)
        class_inter[b]=inter_datas[0]
        bbbox_inter[b]=inter_datas[1]
        landm_inter[b]=inter_datas[2]
    """
    Inter mediate targets are created
    pos samples -> non zero classes
    pos sample are taken in bounding box and keypoints from predictiona dn intermediate target and smooth_l1_loss is used to find the loss
    """ 
    pos_anchors =  class_inter>0
    no_pos=pos_anchors.long().sum()
    pos_anchors =  torch.unsqueeze(pos_anchors,-1)

    pos_anchors_bbbox =  pos_anchors.expand_as(bbbox_pred)
    pos_anchors_landm =  pos_anchors.expand_as(landm_pred)

    pos_bbox_pred = bbbox_pred[pos_anchors_bbbox].view(-1,4)
    pos_bbox_inter=bbbox_inter[pos_anchors_bbbox].view(-1,4)

    pos_landm_pred = landm_pred[pos_anchors_landm].view(-1,self.lm*2)
    pos_landm_inter= landm_inter[pos_anchors_landm].view(-1,self.lm*2)
    

    landm_loss=F.smooth_l1_loss(pos_landm_pred, pos_landm_inter, reduction='sum')
    bbbox_loss=F.smooth_l1_loss(pos_bbox_pred, pos_bbox_inter, reduction='sum')


    box_acc=self.regress_acc(pos_bbox_pred, pos_bbox_inter)
    land_acc=self.regress_acc(pos_landm_pred, pos_landm_inter)


    """
    hard mining
    change shape to batchsize*no_Anc,no_Class  anchors considered as batches
    find loss for prediction and tru value
    store positive loss( anchors whiche are positive)
    copy loss to negative loss and remove the poitive loss
    sort the loss 
    take the top neg(no pos* ratio) loss

    sum the positive and new negative loss
    """
    [bz,no_Anc]=class_inter.shape

    class_pred = class_pred.view(-1, 2)
    class_inter = class_inter.view(-1)
    class_acc=self.accuracy(class_pred,class_inter)
    
    loss_all_anchors=F.cross_entropy(class_pred,class_inter,reduction='none')
    loss_all_anchors=loss_all_anchors.view(bz,no_Anc)
    #print(loss_all_anchors.shape)
    
    no_neg=no_pos*self.no_neg_ratio

    loss_pos_anchors=loss_all_anchors[pos_anchors.view(bz,no_Anc)]
                                                       
    loss_neg_anchors=loss_all_anchors.clone()   
    loss_neg_anchors[pos_anchors.view(bz,no_Anc)]=0


    loss_neg_anchors, _ = loss_neg_anchors.sort(dim=1, descending=True)  
    hardness_ranks = torch.LongTensor(range(no_Anc)).unsqueeze(0).expand_as(loss_neg_anchors).to(self.device) 
    hard_negatives = hardness_ranks < no_neg     

    loss_neg_anchors = loss_neg_anchors[hard_negatives]  
    class_loss = (loss_neg_anchors.sum() + loss_pos_anchors.sum())
    return class_loss,bbbox_loss,landm_loss,[class_acc,box_acc,land_acc],no_pos  