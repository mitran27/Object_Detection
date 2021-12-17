# Object_Detection
Create various object detection models
and backbones Entirely from scratch using pytorch /tensorflow
like Tradional,Rcnn,faster Rcnn, SSD,Fpn,RetinaNet ,RetinaFace


After building state of the models in classification object detection model started to evolve

<h2>Traditional detection </h2>
<p>
  Traditional object detection uses the concept of <b>Sliding Window </b>and pretrained classification model classifies the cropped image
  </p>
  
<h2>RCNN</h2>
<p>
   Rcnn uses the algorithm<b>Selective search </b> where initial proposals are created using felzenszwalb segmentation and all the proposed region are given to pretrained classifier
  </p>
 
<h2>Fast Rcnn</h2>
<p>
  selective search is used as region proposals , the roi pooling is done on region proposals correspondingly  in the feature maps from the backbone network and the roi's are passed to twin network of object class and bounding box regression to detect the object
  </p>
  
 <h2>Faster Rcnn</h2>
<p>
  RPN is used as region propasal which uses twin network on  the feature maps created with the shared backbone of Faster rcnn 
  rpn is trained with the help of anchors for proving region of anchors 
  anchors are default boxes with different size and scales present in different regions of the input image
  rpn need to find the anchors which has possibility of finding the object and precise location of the object with respect to anchor coordinated
  </p>
