import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import color


folder = '../Corel_Dataset/'
images = np.load(folder + 'images_rgb.npy')
images = np.load(folder + 'images_lab.npy')


image_lab = images[54]
image_rgb = color.lab2rgb(image_lab)

# plt.imshow(image)
# plt.show()


# image_bgr = images[55]

# image_rgb = image_bgr
# image_rgb[:,:,0] = image_bgr[:,:,2]
# image_rgb[:,:,2] = image_bgr[:,:,0]


# plt.imshow(image_bgr)
# plt.show()

small = 3
medium = 5
big = 7

img_blur  = cv2.GaussianBlur(image_lab,(medium,medium),1)

img_hsv     = cv2.GaussianBlur(image_rgb,(medium,medium),1) - cv2.GaussianBlur(image_rgb,(small,small),1)
img_dog     = cv2.GaussianBlur(image_rgb,(big,big),1) - cv2.GaussianBlur(image_rgb,(medium,medium),1)
img_dom     = cv2.GaussianBlur(image_rgb,(big,big),1) - cv2.GaussianBlur(image_rgb,(small,small),1)

# plt.imshow(img)
# plt.show()


# Plot stuff
plt.subplot(2,3,1),plt.imshow(image_rgb)
plt.title('image'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,2),plt.imshow(color.lab2rgb(img_blur))
plt.title('blur'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,3),plt.imshow(color.lab2rgb(img_blur))
plt.title('CIELab'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,4),plt.imshow(img_hsv)
plt.title('M-S'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,5),plt.imshow(img_dog)
plt.title('L-M'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,6),plt.imshow(img_dom)
plt.title('L-S'), plt.xticks([]), plt.yticks([])


plt.show()
