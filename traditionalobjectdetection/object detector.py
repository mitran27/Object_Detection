# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 19:00:56 2020

@author: mitran
"""

import cv2
import tensorflow as tf
from tensorflow.keras.applications import vgg16
import numpy as np

# Image pyramids

def pyramid(image,scale,min_size,gaussian=None):
    
    curr_img_size=image.shape
    ch,cw,_=(curr_img_size)
    nh,mw=(min_size)
    features=[]
   
    while(ch>=nh and cw>=mw):
        features.append(image)
        ch,cw=int(ch/scale),int(cw/scale)
        image = cv2.resize(image,(ch, cw))
        
    return features


def classification(image):
    if(len(image.shape)<4):
        image=np.expand_dims(image,0)
        
   
    x=model(image)
    clas=np.argmax(x)
    prob=np.max(x)
    return [clas,prob]
    


    
    
def sliding_Window(image,window_Size,stride,classification=None):
    wh,ww=(window_Size)
    img_h,img_w,_=image.shape
    print(image.shape)
    results=[]
    for i in range(0,img_h-wh,stride):
        for j in range(0,img_w-ww,stride):
            roi=image[i:i+wh,j:j+ww,:]
            
            yield [i,j,classification(roi)]

# Non-Maximum Suppression
#compute the area of the bounding boxes and sort the bounding
# boxes by the bottom-right y-coordinate of the bounding box
def non_max_suppression_fast(boxes, overlapThresh=0.3):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(idxs[-1])
		
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
        
		overlap = (w * h) / area[idxs[:last]]
        
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	
	return boxes[pick].astype("int")
             

"""
1) Image pyramids: Localize objects at different scales/sizes.
2) Sliding windows: Detect exactly where in the image a given object is.
3) Non-maxima suppression: Collapse weak, overlapping bounding boxes.
"""
image=cv2.imread('./carss.jpg')
image=cv2.resize(image,(756,756))
model=tf.keras.applications.VGG19(weights='imagenet')
"""
x=model(image)
x=np.array((x))

#https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
p=np.argmax(x)

#817 is sports car  , it classified correctly
"""

pyramids_features=pyramid(image, 1.5, (224,224))
 


# For each scale of the image pyramid, run a sliding window
# Take the ROI and pass it through our CNN and Examine the probability 
confidence_prob=0.9
od=[]
for i,img in enumerate(pyramids_features):    
    for p in sliding_Window(img,(224,224),8,classification):
        if(p[2][1]>confidence_prob):
           od.append([i,p])

boxes=[]    
image=cv2.imread('./carss.jpg')
image=cv2.resize(image,(756,756))


for i in od:
   scale=1.5**i[0] 
   print(scale)
   x1=(int(i[1][0]*scale),int(i[1][1]*scale))
   x3=(int(i[1][0]*scale+(224*scale)),int((i[1][1]*scale)+(224*scale)))
   cv2.rectangle(image,x1 ,x3 , (0,255,0), 2)
   boxes.append([x1[0],x1[1] ,x3[0],x3[1]])
   
   

cv2.imwrite('imgbeforeNMS.jpg',image)




   
box=np.array((boxes))    
bounding_box=non_max_suppression_fast(box)[0]

image=cv2.imread('./carss.jpg')
image=cv2.resize(image,(756,756))

cv2.rectangle(image,(bounding_box[0],bounding_box[1]) ,(bounding_box[2],bounding_box[3]) , (0,255,0), 2)    
cv2.imwrite('imgaftreNMS.jpg',image)
