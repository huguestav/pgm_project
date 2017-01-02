# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import cv2
import numpy as np

image_name = '710026.jpg'
img = cv2.imread(image_name,0)
sift = cv2.xfeatures2d.SIFT_create(0,3,1E-5,1000,1.6)

# Computes SIFT detector
kp = sift.detect(img, None)
kp,des = sift.compute(img,kp)
#kp, des = sift.detectAndCompute(img,None)

width, height = img.shape[:2]

#Draws the SIFT keypoints
cv2.drawKeypoints(img,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


cv2.imwrite('sift_keypoints2.jpg', img)
cv2.imshow('image', img)
cv2.waitKey(0)

