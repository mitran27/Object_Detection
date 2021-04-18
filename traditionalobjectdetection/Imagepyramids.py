# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 13:40:32 2020

@author: mitran
"""

import cv2
import numpy as np
image=cv2.imread('./car.jpg')
image=cv2.resize(image,(1024,1024))
def pyramid(image,scale,min_size,gaussian=None):
    
    curr_img_size=image.shape[:-1]
    ch,cw=(curr_img_size)
    nh,mw=(min_size)
    images=[]
    while(ch>nh and cw>mw):
        
        images.append(image)
        ch,cw=int(ch/scale),int(cw/scale)
        image = cv2.pyrDown(image)
        
    return images
def pUp(image):
    return cv2.resize(image,(image.shape[0]*2,image.shape[1]*2))
def laplacian_filter(pyramids):
    lf=[]
    for i in range(len(pyramids)-1):# top of img is not taken for difference
    
        lap_filter=cv2.subtract(pyramids[i],cv2.pyrUp(pyramids[i+1]))
        lf.append(lap_filter)
    lf.append(pyramids[-1])
    return lf
    
        
images=pyramid(image, 2, (16,16),5)
lap_img=laplacian_filter(images)
for m,i in enumerate(lap_img):
    li=np.where(i>2,255,0)
    cv2.imwrite(str(m)+'.jpg',li)
        
        