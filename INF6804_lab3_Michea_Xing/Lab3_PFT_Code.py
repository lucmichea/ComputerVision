# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 14:35:13 2019

@author: Luc & Jin
"""

###############################################################################
####    Imports
###############################################################################

import cv2  
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random as ran
import glob

###############################################################################
####    Definition of the functions
###############################################################################

def CompareParticles(hist1, hist2):
    """
    Define a weight of comparison between a particule's histogram and the model's histogram.
    
    Steps : Basicaly we calculate the distance between two histograms.
    (bhattacharya so it is reprensented as a probability)
    Then we return a value between 0 and 1. 
    The higher the value is, the more probable the histogram is the same.
    """
    return 1 - cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

def CalcHisto(image, bbox):
    """
    Calculate the Histogram of a picture in a certain area (defined by a rectangle).
    
    Steps : We define the region of interest and the mask. 
    Then we calculate the histogram.
    """
    roi = (bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3])
    mask = np.zeros((image.shape[0],image.shape[1]), np.uint8)
    cv2.rectangle(mask,(roi[0],roi[1]),(roi[2],roi[3]),255,-1,8,0);
    return cv2.calcHist([image],[0],mask,[64],[0,256])

def GenererParticules(Particules, nbparticules, mvt = 20, scl = 20): 
    """
    Generate the particules, at random around the object, at random for the scale of the rectangle.

    Commentary : We removed one of the mode, each time we change the bbox to the previous one.  
    """
    NouvParticules = Particules
    for i in range(1,nbparticules):
        part = [(Particules[0][0]+ran.randint(-mvt,mvt), Particules[0][1]+ran.randint(-mvt,mvt), Particules[0][2]+ran.randint(-scl,scl), Particules[0][3]+ran.randint(-scl,scl))]
        NouvParticules = NouvParticules + part
    return NouvParticules

def UpdateModel(org_modele,modele, association, image, mouvement, scale):
    """
    Function with the aim to update the appearence of the object. 
    
    Steps : At first we try to find the best rectangle and then we update.
    """
    particules= GenererParticules([association],15*nbpart, 5*mouvement,2*scale) #We generate the particules

    #We calculate the weight of each particules compared to the original picture.
    poids = []
    poids2 = []
    for p in particules:
        candidat = CalcHisto(image, p)
        dist = CompareParticles(modele, candidat)
        dist2= CompareParticles(org_modele,candidat)
        poids.append(dist)
        poids2.append(dist2)
    #We calculate which rectangle corresponds to the best weight and we plot it          
    p = particules[poids.index(max(poids))]
    p2 = particules[poids2.index(max(poids2))]
    if ((poids.index(max(poids))!=poids2.index(max(poids2)))and((max(poids2)>=0.6)and(p2[2]>edge_lim and p2[3]>edge_lim))): 
        association=p2
        modele=CalcHisto(image,p2)
    else :
        if ((max(poids)>=0.8)and(p[2]>edge_lim and p[3]>edge_lim)):
            association=p
            modele=CalcHisto(image, p)
    return (modele,association)

def deleteContent(pfile):
    """
    Function to delete the content of a file
    """
    pfile.seek(0)
    pfile.truncate()

###############################################################################
####    Import of pictures and parameters
###############################################################################

#filenames = [img for img in glob.glob("C:/Users/lucmi/OneDrive/Documents/INF6804/Datasets/2012/dataset/baseline/PETS2006/input/*.jpg")]
filenames = [img for img in glob.glob("C:/Users/lucmi/OneDrive/Documents/INF6804/Datasets/TP3_data(final)_COPY/frame/*.jpg")]
#filenames = [img for img in glob.glob("C:/Users/lucmi/OneDrive/Documents/INF6804/Datasets/TP3_data(final)/frame/*.jpg")]

filenames.sort() # ADD THIS LINE

imlist = []
for img in filenames:
    n= cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    imlist.append(n)


#parameters of images   
im = imlist[0] #first image
height = np.size(im, 0) 
width = np.size(im, 1)
#RGB = np.size(im, 2)
N = len(imlist)

#We modify the length of imlist at first for tests (complexity) (persons)
start=0 #216 for PETS2006
#N_im=100
#imlist=imlist[start:start+N_im]  
#N = len(imlist)

##We modify the length of imlist at first for tests (complexity) (cups)
#imlist=imlist[0:100]  
#N = len(imlist)

###############################################################################
####    Results file creation
###############################################################################

#We create the file for the results
#result = open("C:/Users/lucmi/OneDrive/Documents/INF6804/Datasets/2012/dataset/baseline/PETS2006/results.txt","w+")
result = open("C:/Users/lucmi/OneDrive/Documents/INF6804/Datasets/TP3_data(final)_COPY/results.txt","w+")
#result = open("C:/Users/lucmi/OneDrive/Documents/INF6804/Datasets/TP3_data(final)/results.txt","w+")

deleteContent(result)
#The output of your tracker should be provided in a 'results.txt' file,
#and should be formatted so that we can read it automatically. We expect
#the following structure for our parser:
#
# <frameid> <cupid> <xmin> <xmax> <ymin> <ymax>
#
#Example: if for the 36th frame of the sequence your method says the cup1 (in black)
#is in the rectangle defined with the two <x,y> points [32,45] and [65,112] and the cup2 (in red) is in the rectangle defined with the two <x,y> points [18,30] and [42,75],
#then the output for that line in 'results.txt' should be:
#
# 36 1 32 65 45 112
# 36 2 18 42 30 75
#
#That file should contain exactly 2022 lines, and the first line should start
#with a frame index of 1. Besides, note that the bounding box provided for
#the initialization of your method in 'init.txt' is formatted the same way.


#exemple de remplissage du fichier.
#for i in range(10):
#     f.write("This is line %d\r\n" % (i+1))

result.write("FrameId \r CupId \r Xmin \r Xmax \r Ymin \r Ymax \n")
result_table=[]

###############################################################################
####    We create the bboxes table and plot them (first ones)
###############################################################################

#init = open("C:/Users/lucmi/OneDrive/Documents/INF6804/Datasets/2012/dataset/baseline/PETS2006/init.txt","r")
init = open("C:/Users/lucmi/OneDrive/Documents/INF6804/Datasets/TP3_data(final)_COPY/init.txt","r")
#init = open("C:/Users/lucmi/OneDrive/Documents/INF6804/Datasets/TP3_data(final)/init.txt","r")

bboxes_init=[]
for line in init:
    B=line.split()
    bboxes_init.append([int(B[2]),int(B[4]),int(B[3])-int(B[2]),int(B[5])-int(B[4])])
    
N_obj=len(bboxes_init)
## We chose which one of the three person we will follow
#bbox = (279, 120, 36, 120)
##bbox = (83, 300, 70, 160)
##bbox = (547, 90, 45, 100)
image1=imlist[0]
fig,ax = plt.subplots(1)
ax.imshow(image1,cmap = plt.get_cmap('gray'))
for i in range(N_obj):
    bbox=bboxes_init[i]
    rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=2,edgecolor='r',facecolor='none')
    result_table.append([1,i+1,bbox[0],bbox[0]+bbox[2],bbox[1],bbox[1]+bbox[3]])
    ax.add_patch(rect)
plt.show()

init.close()


###############################################################################
###############################################################################
####    Begining of the tracking of the objects
###############################################################################
###############################################################################

###############################################################################
####   Initialisation
###############################################################################

nbpart = 100  #Number of particules (rectangles created at random)
mouvement = 150  #Max mouvement on X and Y (in pixels)
scale = 30 #Max change of scale (in pixels)
#dist_lim = 340 #Limit on the distance between two rectangles detected (in pixels)
w_lim=0.65 #Limit in the weight, if a rectangle is below this the rectangle is not believed to be the object
edge_lim=60 #Limit of the length 
upd_nb=0
upd_lim=30 #every 30 pictures, we update the model.
deltamvt=30 #if the object disapears, we can move more.
deltascl=10 #if the object disapears, the rectangle scale can change more.
modele = []
org_modele = []
mvt_list= []
scl_list= []
for o in range(N_obj):
    modele.append(CalcHisto(image1, bboxes_init[o])) #Histogram we will use for the comparison (histogram of the inside of the rectangle)
    org_modele.append(CalcHisto(image1, bboxes_init[o])) #Histogram we will use for the comparison (histogram of the inside of the rectangle)
    mvt_list.append(mouvement)
    scl_list.append(scale)
bboxes=bboxes_init

###############################################################################
####    Loop on the pictures of the file
###############################################################################
    
for i in range(1,N): 
    imagetrack=imlist[i]
    detected = []
    
###############################################################################
####    Loop on the objects we need to track
###############################################################################

    for o in range(N_obj):
        
        bbox=bboxes[o]
        particules= GenererParticules([bbox],nbpart, mvt_list[o],scl_list[o]) #We generate the particules

        #We calculate the weight of each particules compared to the original picture.
        poids = []
        for p in particules:
            candidat = CalcHisto(imagetrack, p)
            dist = CompareParticles(modele[o], candidat) 
            poids.append(dist)
        #We calculate which rectangle corresponds to the best weight and we plot it          
        p = particules[poids.index(max(poids))]
        if ((max(poids)>=w_lim)and(p[2]>edge_lim and p[3]>edge_lim)):
            detected.append([p[0],p[1],p[2],p[3]])
            mvt_list[o]=mouvement
            scl_list[o]=scale
        else:
            mvt_list[o]+=deltamvt
            scl_list[o]+=deltascl
    #We try to find if two rectangles detected were the same
#    d=[]
#    for j in range(len(detected)-1):
#        for k in range(len(detected)):
#            rect1=detected[j]
#            if (k!=j):
#                rect2=detected[k]
#                d.append(abs(rect1[0]-rect2[0])+abs(rect1[1]-rect2[1])+abs(rect1[2]-rect2[2])+abs(rect1[3]-rect2[3]))
#                if (d<=dist_lim):
#                    detected.pop(k)
    
    #We calculate the comparison between every objects detected
    association=[]
    indexleft=[]
    poids_compare=[]
    val1=[]
    val2=[]
    for o in range(N_obj):
        poids = []
        val3=[]
        for j in range(len(detected)):
            candidat = CalcHisto(imagetrack,detected[j])
            dist = CompareParticles(modele[o], candidat)
            poids.append(dist)
            val3.append(dist)
        poids_compare.append(poids)
        val1.append(poids)
        val2.append(val3)
        #we set the values that will be used for the association
        association.append([])
        indexleft.append(o)
        
    
    #We do the association of each objects. 
    poids=[]
    for k in range (len(detected)):
        index1=poids_compare.index(max(poids_compare))
        poids=poids_compare.pop(index1)
        index2=poids.index(max(poids))
        index1=indexleft.pop(index1)
        association[index1]=detected.pop(index2)
        for j in range (len(poids_compare)) :
            poids_compare[j].pop(index2)
    
    #We update the model and we associate bboxes to the association done.
    upd_nb+=1
    for o in range(len(association)):
        if (len(association[o])!=0):
            if(upd_nb>=upd_lim):
                upd_nb=0
                (modele[o],association[o])=UpdateModel(org_modele[o],modele[o],association[o],imagetrack,mouvement,scale)
            bboxes[o]=association[o]
            result_table.append([i+1,o+1,association[o][0],association[o][0]+association[o][2],association[o][1],association[o][1]+association[o][3]])
        else:
            result_table.append([i+1,o+1,None,None,None,None])

x=ran.randint(1,N)
imagetrack=imlist[x]
fig,ax = plt.subplots(1)
ax.imshow(imagetrack,cmap = plt.get_cmap('gray'))
bboxes=result_table[N_obj*x:N_obj*(x+1)]
for o in range(N_obj):
    bbox=bboxes[o][2:6]
    if (bbox[0]!=None):
        bbox=[bbox[0],bbox[2],bbox[1]-bbox[0],bbox[3]-bbox[2]]
        rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=2,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
plt.show()

for r in range (len(result_table)):
    result.write("%s \r %s \r %s \r %s \r %s \r %s \n" %(start-1+result_table[r][0], result_table[r][1], result_table[r][2], result_table[r][3], result_table[r][4], result_table[r][5]))

result.close() 