#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 13:53:47 2019

@author: Luc and Jinling
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import random

#Detection with HOG + SVM
####
## Pictures import
####

#import images
#filenames = [img for img in glob.glob("C:/Users/lucmi/OneDrive/Documents/INF6804/Datasets/VOT2013/Illumination/*.jpg")]
#filenames = [img for img in glob.glob("C:/Users/lucmi/OneDrive/Documents/INF6804/Datasets/VOT2013/Obstruction/*.jpg")]
filenames = [img for img in glob.glob("C:/Users/lucmi/OneDrive/Documents/INF6804/Datasets/VOT2013/Size/*.jpg")]
filenames.sort() # ADD THIS LINE

imlist = []
for img in filenames:
    n= cv2.imread(img)
    imlist.append(n)


#parameters of images   
im = imlist[0] #first image
height = np.size(im, 0) 
width = np.size(im, 1)
RGB = np.size(im, 2)
N = len(imlist)

####
## Initialization of the detector
####

#load cascade classifier training file for lbpcascade

lbp= cv2.CascadeClassifier('C:/Users/lucmi\Anaconda2/pkgs/libopencv-3.4.2-h875b8b8_0/Library/etc/lbpcascades/lbpcascade_frontalface.xml')

rects_list = []
epsilon_h=30   
epsilon_w=30
d_max=200
tracking_ok=0   #represents if the tracking was made ok or not
tracking_error=0    #represents when the tracking failed
rect_obj_list=[]    #contains the rectangles in a picture.        

for i in range(0,N):  
    im = imlist[i]
    im1 = im[:, :, 0]
    rects = lbp.detectMultiScale(im1, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10) )
    if len(rects)==0:
        tracking_error+=1
        
    else:    
        rects_list.append([rects[0,0],rects[0,1],rects[0,2],rects[0,3],i])
    
#        for x, y, w, h in rects:
#            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        rect1=rects_list[-1]
        cv2.rectangle(imlist[i], (rect1[0], rect1[1]), (rect1[0]+rect1[2], rect1[1]+rect1[3]), (255, 255,255), 2)
        #Tracking :
        #Previous object tracked correctly
        if len(rect_obj_list)==0:
#            obj=[99,105,33,33]              #illumination
#            obj=[197,93,52,52]              #obstruction
            obj=[185,73,43,43]              #scale
            
                                            #you need to give approximately 
                                            #the middle of the face of the person on first frame
                                            #give random values at first and then fill it with the first element
                                            #of rects_list
        else:
            obj=rect_obj_list[-1]
            
            
        #Check if tracking correct
        new_mid = [rect1[0]+rect1[2]/2,rect1[1]+rect1[3]/2]
        prev_mid = [obj[0]+obj[2]/2,obj[1]+obj[3]/2]
        
        if ((abs(obj[2]-rect1[2])<epsilon_h)and(abs(obj[3]-rect1[3])<epsilon_w)
                and(abs(new_mid[0]-prev_mid[0])+abs(new_mid[1]-prev_mid[1])<d_max)):
            rect_obj_list.append([rect1[0],rect1[1],rect1[2],rect1[3],i])
            cv2.putText(imlist[i], "Id1", (rect1[0], rect1[1]-5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255) , 2)
            tracking_ok+=1
        else:
             tracking_error+=1
             
             
####
##  evaluation criteria
####
        
tracking_complete=float(tracking_ok)/N
tracking_uncomplete=float(tracking_error)/N
tracking_obj_detected=float(len(rect_obj_list))/len(rects_list)    

####
#   affichage
####
    
a=102
plt.figure(figsize = (8,8))
plt.imshow(imlist[a], cmap = plt.get_cmap('gray'))
plt.show()
plt.figure(figsize = (8,8))
plt.imshow(imlist[a+1], cmap = plt.get_cmap('gray'))
plt.show()    
plt.figure(figsize = (8,8))
plt.imshow(imlist[a+2], cmap = plt.get_cmap('gray'))
plt.show()    
#
#start = 3 #find a fram with only one rectangle that have the subject inisde
#rect_courant=rects_list[start]
#if (len(rect_courant)!=1):
#    tracking_error=1
#else:
#    rect_obj=[rect_courant[0,0],rect_courant[0,1],rect_courant[0,2],rect_courant[0,3],0]
#    rect_obj_list.append(rect_obj)
#          
#for i in range(start+1,100):
#    rect_new=rects_list[i]
#    if (len(rect_new)==0):
#        tracking_error+=1
#    else:
#        obj=rect_obj_list[-1]
#        mid1=[obj[0]+obj[2]/2,obj[1]+obj[3]/2]
#        for j in range (0, len(rect_new)):
#            nearestmid=200000
#            test=rect_new[j]
#            mid2=[test[0]+test[2]/2,test[1]+test[3]/2]
#            if(abs(mid1[0]-mid2[0])+abs(mid1[1]-mid2[1])<nearestmid):
#                nearest_rect=rect_new[j]
#                nearestmid=(abs(mid1[0]-mid2[0])+abs(mid1[1]-mid2[1]))
#            test=nearest_rect
#            mid2=[test[0]+test[2]/2,test[1]+test[3]/2]
#            if ((abs(obj[2]-test[2])<epsilon_h)and(abs(obj[3]-test[3])<epsilon_w)and(abs(mid1[0]-mid2[0])+abs(mid1[1]-mid2[1])<d_max)):
#                 rect_obj_list.append([test[0],test[1],test[2],test[3],i])
#                 cv2.putText(imlist[i], "Id1", (test[0], test[1]-5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#                 tracking_ok+=1
#            else:
#                 tracking_error+=1
                 
                 