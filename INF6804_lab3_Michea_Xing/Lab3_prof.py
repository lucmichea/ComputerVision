# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:06:44 2019

@author: Luc & Jin 
"""

import cv2  
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random as ran


def CompareParticles(hist1, hist2):
    return 1 - cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

def CalcHisto(image, bbox):
    roi = (bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3])
    mask = np.zeros((image.shape[0],image.shape[1]), np.uint8)
    cv2.rectangle(mask,(roi[0],roi[1]),(roi[2],roi[3]),255,-1,8,0);
    return cv2.calcHist([image],[0],mask,[64],[0,256])

def GenererParticules(Particules, nbparticules, mvt = 20, scl = 20, poids = None):
    if len(Particules)==1:
        #Pour générer les particules initiales. Pas de poids disponible
        #Déplacement aléatoire de ROI initial de [-mvt, mvt]
        NouvParticules = Particules
        for i in range(1,nbparticules):
            part = [(Particules[0][0]+ran.randint(-mvt,mvt), Particules[0][1]+ran.randint(-mvt,mvt), Particules[0][2]+ran.randint(-scl,scl), Particules[0][3]+ran.randint(-scl,scl))]
            NouvParticules = NouvParticules + part
    else:
        # Échantillonage préférentiel avec la fonction ran.choices()
        temp = ran.choices(Particules,poids)[0]
        # Mise à jour de leur état en ajoutant une translation en X et Y.
        part = [(temp[0]+ran.randint(-mvt,mvt), temp[1]+ran.randint(-mvt,mvt), temp[2]+ran.randint(-scl,scl), temp[3]+ran.randint(-scl,scl))]
        NouvParticules = part
        for i in range(1,nbparticules):
            temp = ran.choices(Particules,poids)[0]
            part = [(temp[0]+ran.randint(-mvt,mvt), temp[1]+ran.randint(-mvt,mvt), temp[2]+ran.randint(-scl,scl), temp[3]+ran.randint(-scl,scl))]
            NouvParticules = NouvParticules + part
    return NouvParticules


# We import the pictures and we plot them
image1 = cv2.imread('C:/Users/lucmi/OneDrive/Documents/INF6804/Datasets/2012/dataset/baseline/PETS2006/input/in000216.jpg',cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('C:/Users/lucmi/OneDrive/Documents/INF6804/Datasets/2012/dataset/baseline/PETS2006/input/in000219.jpg',cv2.IMREAD_GRAYSCALE) 
f, axarr = plt.subplots(1, 2, figsize=(15,15))
axarr[0].imshow(image1,cmap = plt.get_cmap('gray'))
axarr[1].imshow(image2,cmap = plt.get_cmap('gray'))
plt.show()


# We chose which one of the three person we will follow
bbox = (279, 120, 36, 120)
#bbox = (83, 300, 70, 160)
#bbox = (547, 90, 45, 100)
fig,ax = plt.subplots(1)
ax.imshow(image1,cmap = plt.get_cmap('gray'))
rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=2,edgecolor='r',facecolor='none')
ax.add_patch(rect)
plt.show()




nbpart = 80  #Number of particules (rectangles created at random)
mouvement = 40  #Max mouvement on X and Y (in pixels)
modele = CalcHisto(image1, bbox) #Histogram we will use for the comparison (histogram of the inside of the rectangle)
particules= GenererParticules([bbox],nbpart, mouvement) #We generate the particules
fig,ax = plt.subplots(1,figsize=(10,10)) #We plot the picture with all the particules (rectangles)
ax.imshow(image1,cmap = plt.get_cmap('gray'))
for p in particules:
    rect = patches.Rectangle((p[0],p[1]),p[2],p[3],linewidth=2,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
plt.show()

#We calculate the weight of each particules compared to the original picture.
poids =[]
for p in particules:
    candidat = CalcHisto(image2, p)
    dist = CompareParticles(modele, candidat) 
    poids.append(dist)
    
#We calculate which rectangle corresponds to the best weight and we plot it
fig,ax = plt.subplots(1)
ax.imshow(image2,cmap = plt.get_cmap('gray'))
p = particules[poids.index(max(poids))]
rect = patches.Rectangle((p[0],p[1]),p[2],p[3],linewidth=2,edgecolor='r',facecolor='none')
ax.add_patch(rect)
plt.show()



###################################################
#   We do the same steps but with two new pictures.
###################################################
scale= 20;
image3 = cv2.imread('C:/Users/lucmi/OneDrive/Documents/INF6804/Datasets/2012/dataset/baseline/PETS2006/input/in000222.jpg',cv2.IMREAD_GRAYSCALE) 
particules = GenererParticules(particules,nbpart, mouvement, scale , poids)
poids =[]
for p in particules:
    candidat = CalcHisto(image3, p)
    dist = CompareParticles(modele, candidat)
    poids.append(dist)
    
fig,ax = plt.subplots(1)
ax.imshow(image3,cmap = plt.get_cmap('gray'))
p = particules[poids.index(max(poids))]
rect = patches.Rectangle((p[0],p[1]),p[2],p[3],linewidth=2,edgecolor='r',facecolor='none')
ax.add_patch(rect)
plt.show()

##############################################################################

image4 = cv2.imread('C:/Users/lucmi/OneDrive/Documents/INF6804/Datasets/2012/dataset/baseline/PETS2006/input/in000225.jpg',cv2.IMREAD_GRAYSCALE) 
particules = GenererParticules(particules, nbpart, mouvement, scale , poids)
poids =[]
for p in particules:
    candidat = CalcHisto(image4, p)
    dist = CompareParticles(modele, candidat)
    poids.append(dist)
    
fig,ax = plt.subplots(1)
ax.imshow(image4,cmap = plt.get_cmap('gray'))
p = particules[poids.index(max(poids))]
rect = patches.Rectangle((p[0],p[1]),p[2],p[3],linewidth=2,edgecolor='r',facecolor='none')
ax.add_patch(rect)
plt.show()