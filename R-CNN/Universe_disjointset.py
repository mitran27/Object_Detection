# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 19:08:37 2020

@author: mitran
"""
import numpy as np
# disjoint-set forests using union-by-rank and path compression

"""there are a many element in the universe elements can group themselves into sub groups with a constraint that there should be no 
common element in any two groups
Applications : image segmentation
               friends Circle
"""
class Universe(): #disjoint set
    """
    Inputs : no elements 
    Output : the set with no common elements
    Function: 1) join two elements
              2) Find the group in which the given element is present
    """
    def __init__(self,no_elements):
        self.N=no_elements # initial number of set in universe where each of them are individual set
        # parameter needed are Parent (for finding the group) Height (finding the rank of the two elemets) 
        self.Set=np.zeros((no_elements,3),dtype='int32')
        for i in range(no_elements):
          self.Set[i,0]=0# initially rank of elements are 0
          self.Set[i,1]=i# in ds set initially parent of each elemet is itself because each of them are individual set
          self.Set[i,2]=1 # initially size of the set is 1
    def __len__(self):
        return self.N # return no of segments present
    
    def rank(self,x):
        return self.Set[x,0]
    
    def size(self,x):
        return self.Set[x,2]
        
    def parent(self,x):
        return self.Set[x,1]
    def find(self,x):# finding the rep finds the rooth node and returns but path compression finds the root node of given x and makes the root node as paretn fo x so the 
        self.Set[x,1]=self.rep(x)
        #https://stackoverflow.com/questions/25317156/why-path-compression-doesnt-change-rank-in-unionfind
        return self.parent(x)
    def rep(self,x): # if tree size increases rep will be the root node of the tree
    
        if(self.parent(x)==x):#the root node only will have parent as itself
              return x
        self.Set[x,1]= self.rep(self.parent(x)) # pass the  parent node
        return self.Set[x,1]
    def union(self,x,y): # union is basically making x as a parent of y ( rank of x changes)
    # In union by rank the set having higher rank is made as parent of the one with lower rank( rank do not changes) (rank decide the parent)
    # the rank is height of the tree
    # if both are equal anything is fine (lets prefer x as parent)(rank changes)
    
    # union for already joint elements is not going to happen
       repx=self.rep(x)
       repy=self.rep(y)
       if(repx==repy):
           return 0
       #new set of elements
       self.N-=1
       if(self.rank(repx)>self.rank(repy)):
           pass
           # repy becomes child of repx
           # rank is same
           # size of repx is increased by size of repy
           
           self.Set[repy,1]=repx
           self.Set[repx,2]+=self.size(repy)
           
       elif(self.rank(repx)<self.rank(repy)):
           pass
           # repx becomes child of repy
           # rank is same
           # size of repy is increased by size of repx
           self.Set[repy,2]+=self.size(repx)
           self.Set[repx,1]=repy
       else:# both are same
           pass
           # repy becomes child of repx
           # rank is increased by 1
           # size of repx is increased by size of repy
       
           self.Set[repy,1]=repx
           self.Set[repx,2]+=self.size(repy)
           self.Set[repx,0]+=1
           
       return 1
   
    
           


           
           
           
           
     
     
         
     
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        