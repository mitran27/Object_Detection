# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 17:43:08 2020

@author: mitran
"""
import numpy as np
from math import exp,sqrt
from Universe_disjointset import Universe
import random
from sklearn.preprocessing import normalize
from skimage.filters import gaussian
from scipy.ndimage.filters import convolve

def square(x):
    return x*x     
def random_rgb():
    rgb = np.zeros(3, dtype=int)
    rgb[0] = random.randint(0, 255)
    rgb[1] = random.randint(0, 255)
    rgb[2] = random.randint(0, 255)
    return rgb



class Felzenszwalb():
    def smooth(self,image,sigma=1,kernel=(5,5)):
        return cv2.GaussianBlur(image,(5,5),sigma)
    """
    image(width, height, 3) or (width, height) ndarray
        Input image.

    scalefloat
         Free parameter. Higher means larger clusters.

     sigmafloat
          Width (standard deviation) of Gaussian kernel used in preprocessing.

     min_sizeint
             Minimum component size
    """
    
    def __init__(self,image,sigma,thresh):
        self.h,self.w,self.c=image.shape
        self.K=thresh
        self.C={}
        self.sigma=sigma
        self.C['R']=self.smooth(image[:,:,2]).astype("float32")
        self.C['G']=self.smooth(image[:,:,1]).astype("float32")
        self.C['B']=self.smooth(image[:,:,0]).astype("float32")
        edges_size = self.h*self.w * 4
        edges = np.zeros(shape=(edges_size, 3),dtype='int32')
        wt=0 # weigt number
        
        #Building graph#
        """ the graph will be built in the following manner
        edge[1] has the source position
        edge[2] has the target position
        edge[3] has the cosine distance
        
        [  0  0  T
           0  S  T       
           0  T  T]
        each pixel in the image is considered as a source (S) and its corresponding 
        Targets(T) {
             >> upper right
             >> right
             >> down
             >>lower right
            
            } along with its cosine distance are taken 
        so weight are no of pixels x 4
        
        """
        for y in range(self.h):
            for x in range(self.w):
                # iterating row by row
                
                if(x<self.w-1):
                   edges[wt,0]=(self.w*y+x)
                   edges[wt,1]=(self.w*y+(x+1))
                   edges[wt,2]=self.find_dissimilarity(x,y,(x+1),y)
                   wt+=1
                if(y<self.h-1):
                   edges[wt,0]=(self.w*y+x)
                   edges[wt,1]=(self.w*(y+1)+x)
                   edges[wt,2]=self.find_dissimilarity(x,y,x,(y+1))
                   wt+=1
                if(x<self.w-1 and y<self.h-1):
                   edges[wt,0]=(self.w*y+x)
                   edges[wt,1]=(self.w*(y+1)+(x+1))
                   edges[wt,2]=self.find_dissimilarity(x,y,(x+1),(y+1))
                   wt+=1
                if(x<self.w-1 and y>0):
                   edges[wt,0]=(self.w*y+x)
                   edges[wt,1]=(self.w*(y-1)+(x+1))
                   edges[wt,2]=self.find_dissimilarity(x,y,(x+1),(y-1))
                   wt+=1
        self.edges=edges
        self.edge_cnt=wt
    def thresh_size(self,x):
        """we use a threshold function based on the size of the component,
        τ (C) = k/|C| 
       where |C| denotes the size of C, and k is some constant parameter  """
        return self.K/x             
    def segement_graph(self,min_size,get_rep=True):
        """
        Input:
            no of pixels
            no of edges
            parameters of edges(source , dest,weight[dissimilarity between source and dest])
            threshold function
        Output :
            dis-joint set repesenting the segmentation
            What is a Disjoint Set? A pair of sets which does not have any common element are called disjoint sets.
        
        """
        
        #step 1 Sort E into π = (o1, . . . , om), by non-decreasing edge weight
        
        Edges= self.edges[self.edges[0:self.edge_cnt, 2].argsort()]
        
        # Start with a segmentation S0, where each vertex(pixel) vi is in its own component.
        # creating a initial disjoint set
        sets=Universe(self.h*self.w) # disjoint set datastructure
        
        #Start with a segmentation S0
        
        
        """
        #. Repeat step 3 for q = 1, . . . , m(number of edges.) in sorted edges
        STEP :-3
        * Let vi(source) and vj(target) denote the vertices connected by the q-th edge(Edge having source as vi and target as vj) in the ordering
        oq = (vi,vj )
        If vi and vj are in disjoint set(different set with no common elements)
        
        in q−1(upto previous segmentation) and w(oq) is small compared to the internal difference of both those components, then merge the two components otherwise do nothing(they are not made as a same set)
        
        More formally,
        
        If Cq−1i not equal to Cq−1j and [ w(oq) ≤ MInt(Cq−1i, Cq−1j) then Sqis obtained from Sq−1 by merging Cq−1i and Cq−1j
        . Otherwise S q = Sq−1
        
        if they are not in same segment untill now and theri weight is less than internal difference( a threshhold weight)  then they are merged in new segments list
        
       
        
       MInt(C1, C2) = min(Int(C1) + τ (C1), Int(C2) + τ (C2)).
       
       The threshold function τ controls the degree to which the difference between two
       components(weight) must be greater than their internal differences in order for there to be
       evidence of a boundary between them(no merging)
       
       
       Therefore, 
        
        """
        # initializing Thresholds
        # each pixel component will have its  threshold based on its current size
        pixthres=np.ones(self.h*self.w, dtype=float)*self.thresh_size(1)# at initial the size of each pixel group is 1 (they alone)
        
        # start merging
        
        for i in range(self.edge_cnt):
            src_rep=sets.rep(Edges[i,0])
            tar_rep=sets.rep(Edges[i,1])
            weight=Edges[i,2]
            
            if(src_rep!=tar_rep):

                if(weight<=min(pixthres[src_rep] , pixthres[tar_rep])):
                    sets.union(src_rep,tar_rep)
                    new_rep=sets.rep(src_rep)
                    
                    pixthres[new_rep]=weight+self.thresh_size(sets.size(new_rep))
        x=np.zeros(self.h*self.w,dtype='int32') 
        
        for i in range(self.edge_cnt):
                  src_rep=sets.rep(Edges[i,0])
                  tar_rep=sets.rep(Edges[i,1])
                  if((sets.size(src_rep)<min_size or sets.size(tar_rep)<min_size) and (src_rep!=tar_rep)):
                      sets.union(src_rep,tar_rep)
                
                
        for i in range(self.h*self.w):
            x[i]=sets.rep(i)
        if(get_rep==True):  
            return x.reshape(self.h,self.w),sets  
            
        segments=list(set(x))
        
        segmap={s:i for i,s in enumerate(sorted(segments),start=0)}
        s=np.zeros(self.h*self.w,dtype='int32') 
        
        for i in range(self.h*self.w):
            s[i]=segmap[x[i]]
        return s.reshape(self.h,self.w),sets  
             
             
    def find_dissimilarity(self, x1, y1, x2, y2): #similar near pixels low weight dissimilar pixel heigh weight
        
        x = sqrt(square(self.C['R'][y1, x1] - self.C['R'][y2, x2]) + square(self.C['G'][y1, x1] - self.C['G'][y2, x2]) + square(self.C['B'][y1, x1] - self.C['B'][y2, x2]))
        return x
                 
                
        
            
class colour_segments():
    
    def __init__(self,segments):
        self.seg_Col={i:random_rgb() for i in segments}
    def __call__(self,image,segmented_set,name):    
        
        
        #seg_Col={i:img[i,:] for i in segments}
        h,w=image.shape[:-1]
        segmented_set=segmented_set.reshape(-1)
        segmented_image=np.zeros((h*w,3),dtype="uint8")
        for i in range(h*w):
            segmented_image[i,:]=self.seg_Col[segmented_set[i]]
                                                 
        
        
        segmented_image=segmented_image.reshape((h,w,3))
        
        cv2.imwrite(name,segmented_image)


def color_hist(image,segment,nbins=25):
    bins=np.linspace(0,255,nbins+1)
    no_seg=len(set(segment.reshape(-1)))
    labels=[i for i in range(1,2+no_seg)]
    bins=[labels,bins]
    n_channels=3
    histogram=np.hstack([np.histogram2d(segment.reshape(-1),image[:,:,i].reshape(-1),bins=bins)[0] for i in range(n_channels)])# stacking the color channels after creating histogrmas for each of these 
    return normalize(histogram,norm='l1',axis=1)
def texture_hist(img,segment_mask,n_orientation = 8, n_bins = 10):
	''' 
	Computes texture histograms for all the blobs
	parameters
	----------
	img : Input Image
	segment_ mask :  Integer mask indicating segment labels of an image
	returns
	-------
	
	hist : texture histogram of the blobs. Shape: [ n_segments , n_bins*n_orientations*n_color_channels ]
	'''
	filt_img = gaussian(img, sigma = 1.0, multichannel = True).astype(np.float32)
	op = np.array([[-1.0, 0.0, 1.0]])
	grad_x = np.array([convolve(filt_img[:,:,i], op) for i in range(img.shape[-1])])
	grad_y = np.array([convolve(filt_img[:,:,i], op.T) for i in range(img.shape[-1])])
	_theta = np.arctan2(grad_x, grad_y)
	theta = np.zeros(img.shape)
	for i in range(img.shape[-1]):theta[:,:,i] = _theta[i]
	n_segments = len(set(segment_mask.flatten()))
	labels = range(n_segments + 1)	
	bins_orientation = np.linspace(-np.pi, np.pi, n_orientation + 1)
	bins_intensity = np.linspace(0.0, 1.0, n_bins + 1)
	bins = [labels, bins_orientation, bins_intensity]
	_temp = [ np.vstack([segment_mask.flatten(), theta[:,:,i].flatten(), filt_img[:,:,i].flatten()]).T for i in range(img.shape[-1])]
	hist = np.hstack([ np.histogramdd(_temp[i], bins = bins)[0] for i in range(img.shape[-1]) ])
	hist = np.reshape(hist,(n_segments,n_orientation*n_bins*img.shape[-1]))
	hist = normalize(hist,norm='l1',axis=1)
	return hist
    
 
        
import cv2
image=cv2.imread('./dogcat.jpg')
  
z=Felzenszwalb(image,1,100)

segmented_set,sets=z.segement_graph(200,get_rep=False)
#segmented_set = felzenszwalb(image, scale=3.0, sigma=0.8, min_size=100)

segments=list(set(segmented_set.reshape(-1)))

drawsegments= colour_segments(segments)     


drawsegments(image, segmented_set.copy(), 'initialregion.jpg')

from scipy.ndimage import find_objects



# initial regions

colr_hist=color_hist(image, segmented_set)
text_hist=texture_hist(image,segmented_set)

region_property={}

def boxsize(box):
    if((box[2]-box[0]<0) or (box[3]-box[1]<0)):
        return abs(box[2]-box[0])*(box[3]-box[1])*-1
    return (box[2]-box[0])*(box[3]-box[1])
for label in segments:
    
    size=(segmented_set==label).sum()
    region=find_objects(segmented_set==label)[0] 
    boxes=(region[1].start,region[0].start,region[1].stop,region[0].stop)
    bz=boxsize(boxes)
   
       
    region_property[label]={'color':colr_hist[label],
                   'texture':text_hist[label],
                   'size':size,
                   'fill':boxes
        }
# build pairs 


similarities={}

from skimage.segmentation import find_boundaries

def gk(x,y): # get unique key for two regions
    if(x==y):
        print("similarity cannot be found for same two region")
        return None
    return (min(x,y),max(x,y))
def get_neighbours(segmented_set,label):
    boundary = find_boundaries(segmented_set == label,mode='outer')
    neighbours=np.unique(segmented_set[boundary]).tolist()
    
    return neighbours


def get_Similarity(prop_X,prop_Y,image_size):
    
    
    def color(ri,rj):
        #scolour(ri,rj) =n  ∑  k=1 min(cki,ckj)  
        simcol=sum([min(cki,ckj) for cki,ckj in zip(ri,rj) ])
        return simcol
        
    def texture(ri,rj):
        #stexture(ri,rj) =n  ∑  k=1 min(cki,ckj)  
        simtex=sum([min(cki,ckj) for cki,ckj in zip(ri,rj) ])
        return simtex
    
    def  size(ri,rj,imsize):
        #ssize(ri,rj) = 1−(size(ri)+size(rj))/size(im)
        simsize=1-((ri+rj)/imsize)
        return simsize
    def fill(ribox,rjbox,risize,rjsize,imsize):
        
        #fill(ri,rj) = 1−size(BBi j)−size(ri)−size(ri)/size(im)
        bbsize = (max(ribox[2], rjbox[2]) - min(ribox[0], rjbox[0])) * (max(ribox[3], rjbox[3]) - min(ribox[1], rjbox[1]))
        simfil= 1.0 - (bbsize - risize- rjsize) / imsize
        return simfil
    
    
    s=color(prop_X['color'],prop_Y['color'])
    s+=texture(prop_X['texture'],prop_Y['texture'])
    s+=fill(prop_X['fill'],prop_Y['fill'],prop_X['size'],prop_Y['size'],image_size)
    s+=size(prop_X['size'],prop_Y['size'],image_size)
    
    return s

        
        
         
    
    return 1
image_size=segmented_set.shape[0]*segmented_set.shape[1]
for label in segments:
    
    neigbours=get_neighbours(segmented_set,label)
    for nei in neigbours:
        similarities[gk(label,nei)]=get_Similarity(region_property[label],region_property[nei],image_size)
        
    





"""


while S != Null do
Get highest similarity s(ri,rj) = max(S)
Merge corresponding regions rt = ri ∪rj
Remove similarities regarding ri: S = S \ s(ri,r∗)
Remove similarities regarding rj: S = S \ s(r∗,rj)
Calculate similarity set St between rt and its neighbours

"""
def merge_similarity(prop_X,prop_Y):
    
    new_size = prop_X['size'] + prop_Y['size']
    new_box = (min(prop_X['fill'][0], prop_Y['fill'][0]),
                  min(prop_X['fill'][1], prop_Y['fill'][1]),
                  max(prop_X['fill'][2], prop_Y['fill'][2]),
                  max(prop_X['fill'][3], prop_Y['fill'][3]))
    #Ct = size(ri)×Ci +size(rj)×Cjsize(ri)+size(rj)
    
    new_color=( prop_X['size']*prop_X['color']+prop_Y['size']*prop_Y['color'])/new_size
    new_texture=( prop_X['size']*prop_X['texture']+prop_Y['size']*prop_Y['texture'])/new_size
    
    return {'color':new_color,
                   'texture':new_texture,
                   'size':new_size,
                   'fill':new_box
        }
    
    
    
    
def get_highest_Similarity(similarity_Dict):
    assert(type(similarity_Dict)==dict)
    maxsim=0
    bstpair=None
    for k,v in similarity_Dict.items():
        if(v>maxsim):
            maxsim=v
            bstpair=k
    return bstpair  


        
segments=list(set(segmented_set.reshape(-1)))



proposedregions=[]

total=len(segments)
while(len(segments)>1):
    
    
    # get highest priority
    highestsim=get_highest_Similarity(similarities)
    
    #merege regions
    segmented_set=np.where(segmented_set==highestsim[1],highestsim[0],segmented_set)
    segments=list(set(segmented_set.reshape(-1)))
    drawsegments(image, segmented_set,'./test/'+str(total- len(segments))+'.jpg')
    
    # remove ri and rj similarities
    rirjsim=[]
    for rirj in similarities.keys():
        if((highestsim[0] in rirj) or (highestsim[1] in rirj) ):
            rirjsim.append(rirj)
            
    for key in rirjsim:
            del similarities[key]       
            
      
            
    #Calculate similarity set St between rt and its neighbours
    new_region=  highestsim[0]
    
    
    proposedregions.append(region_property[highestsim[0]]['fill'])
    proposedregions.append(region_property[highestsim[1]]['fill'])
    
    
    region_property[new_region]=merge_similarity(region_property[highestsim[0]],region_property[highestsim[1]])
    
    
    # its neighbours
    
    neigbours=get_neighbours(segmented_set,new_region)
    for nei in neigbours:
        similarities[gk(new_region,nei)]=get_Similarity(region_property[new_region],region_property[nei],image_size)
    print(len(segments),len(similarities.keys()))






















    

cv2.imwrite('image.jpg',image)    



  
    
    
from tensorflow.keras.applications import VGG19
model=VGG19(weights='imagenet')


def classification(image):
    image=cv2.resize(image,(224,224))
    if(len(image.shape)<4):
        image=np.expand_dims(image,0)
       
    
    x=model.predict(image)
    clas=np.argmax(x)
    prob=np.max(x)
    return [clas,prob]


    
    
output=[]
confidenceprob=0.6

for box in proposedregions:
     bz=boxsize(box)
     if(bz<500):
         continue
   
     possibleregion=image[box[1]:box[3],box[0]:box[2],:]
     pred=classification(possibleregion)
     if(pred[1]>confidenceprob):
          output.append([pred,box])
          
          
import json


image=cv2.imread('./dogcat.jpg')


with open("labels.json", "r") as read_file:
    label = json.load(read_file)
         
          
for o in output :
     [clas,score],box=o     
     image = cv2.rectangle(image, (box[0],box[1]), (box[2],box[3]), (0,255,0),2)
     objec=label[str(clas)]
     text=str(objec)+'  '+str(score)
     font = cv2.FONT_HERSHEY_SIMPLEX
     image=cv2.putText(image, text, (box[0],box[1]), font, 1, (255,255,0), 1, cv2.LINE_AA, False)
    


               
cv2.imwrite('result.jpg',image)    


import cv2
import numpy as np
import glob

img_array = []
for filename in glob.glob('./test/*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)


out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()    



















































