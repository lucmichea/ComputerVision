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
from skimage.feature import hog
import dlib

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
    n= cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)
#    n= dlib.load_rgb_image(img)
    imlist.append(n)


#parameters of images   
im = imlist[0] #first image
height = np.size(im, 0) 
width = np.size(im, 1)
#RGB = np.size(im, 2)
N = len(imlist)

####
## Initialization of the detector
####

hogFaceDetector = dlib.get_frontal_face_detector()
#faceRects = hogFaceDetector(frameDlibHogSmall, 0)
#hog = cv2.HOGDescriptor()  #64 x 128 pixels
#hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#Multi-scale detection and display.
rects_list = []
tracking_ok=0   #represents if the tracking was made ok or not
tracking_error=0    #represents when the tracking failed
epsilon_h=30   
epsilon_w=30
d_max=60
#middle_init=[125,100]   #you need to give approximately 
#                        #the middle of the face of the person on first frame
rect_obj_list=[]    #contains the rectangles of the tracking in the pictures.   


for i in range(0,N):   
    im = imlist[i]
    im1 = im
    rects = hogFaceDetector(im, 1)
    
    if len(rects)==0:
        tracking_error+=1
        
    else:

        for faceRect in rects:
            x1 = faceRect.left()
            y1 = faceRect.top()
            x2 = faceRect.right()
            y2 = faceRect.bottom()
            rects_list.append([x1,y1,x2,y2,i]) 
        #Trace the white rectangle on the picture
        rect1=rects_list[-1]
        cv2.rectangle(imlist[i], (x1, y1), (x2, y2),  255, 2)
        #Tracking :
        #Previous object tracked correctly
        if len(rect_obj_list)==0:
#            obj=[80,100,120,140]        #illumination
#            obj=[203,98,247,141,0]      #obstruction
            obj=[201,34,237,70]         #scale
            
            
            
                                            #you need to give approximately 
                                            #the middle of the face of the person on first frame
                                            #give random values at first and then fill it with the first element
                                            #of rects_list
        else:
            obj=rect_obj_list[-1]
            
            
        #Check if tracking correct
        new_mid = [(x1+x2)/2,(y1+y2)/2]
        prev_mid = [(obj[0]+obj[2])/2,(obj[1]+obj[3])/2]
        if ((abs((obj[3]-obj[1])-(rect1[3]-rect1[1]))<epsilon_h)and(abs((obj[2]-obj[0])-(rect1[2]-rect1[0]))<epsilon_w)
                and(abs(new_mid[0]-prev_mid[0])+abs(new_mid[1]-prev_mid[1])<d_max)):
            rect_obj_list.append([rect1[0],rect1[1],rect1[2],rect1[3],i])
            cv2.putText(imlist[i], "Id1", (rect1[0], rect1[1]-5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
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
    
a=112
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
#imlist_crit_inter=[]
#imlist_crit_union=[]
#crit_result=[]
#for p in range(0,20):
#    inter_im_test = np.zeros((height, width), np.int)
#    union_im_test = np.zeros((height, width), np.int)
#    im1=imlist_out[p]
#    im2=imlist_gt[p] 
#    sum_inter=0.
#    sum_union=0.
#    for j in range(0, height):
#        for k in range(0, width):
#            inter_im_test[j,k] = im1[j,k]&im2[j,k]
#            union_im_test[j,k] = im1[j,k]|im2[j,k]
#            sum_inter=sum_inter + inter_im_test[j,k]
#            sum_union=sum_union + union_im_test[j,k]
#    crit_result.append(sum_inter/sum_union)
#    imlist_crit_inter.append(inter_im_test)  
#    imlist_crit_union.append(union_im_test)


