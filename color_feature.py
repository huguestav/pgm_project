import cv2
import numpy as np
from matplotlib import pyplot as plt


src = "Corel_Dataset/Images/corel_55.jpg"

img		= cv2.imread(src)

img_rgb		= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_cielab	= cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

img_hsv		= cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

img_dog		= cv2.GaussianBlur(img,(9,9),1) - cv2.GaussianBlur(img,(5,5),1)

img_dom		= cv2.medianBlur(img,9) - cv2.medianBlur(img,5)

plt.subplot(2,3,1),plt.imshow(img)
plt.title('BGR'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,2),plt.imshow(img_rgb)
plt.title('RGB'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,3),plt.imshow(img_cielab)
plt.title('CIELab'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,4),plt.imshow(img_hsv)
plt.title('HSV'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,5),plt.imshow(img_dog)
plt.title('DOG'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,6),plt.imshow(img_dom)
plt.title('DOM'), plt.xticks([]), plt.yticks([])


plt.show()
