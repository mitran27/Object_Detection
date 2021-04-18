# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 13:40:28 2020

@author: mitran
"""
def classify(image):
    return image[0][0]
def sliding_Window(image,window_Size,stride,classification=None):
    wh,ww=(window_Size)
    img_h,img_w=image.shape
    results=[]
    for i in range(0,img_h-wh,stride):
        for j in range(0,img_w-ww,stride):
            roi=image[i:i+wh,j:j+ww]
            results.append([i,j,classification(roi)])
    return results

import numpy as np
image=np.random.randint(0,10,(27,27))

y=sliding_Window(image,(3,3),2,classify)
            
    
    