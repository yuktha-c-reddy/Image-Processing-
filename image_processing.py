import pandas as pd
import numpy as np
from glob import glob
import cv2
import matplotlib.pylab as plt
import os

cats=glob('/content/imageprocessing/cats/*.jpg')
dogs=glob('/content/imageprocessing/dogs/*.jpg')
dog=glob('/content/dog2.jpg')

img_mpl =plt.imread(cats[0])
img_cv =cv2.imread(cats[0])

pd.Series(img_mpl.flatten()).plot(kind="hist",bins=50,title="Pixels")
plt.show()

fig , ax = plt.subplots(figsize=(10,10))
ax.imshow(img_mpl)
ax.axis('off')
plt.show()

fig , axs =plt.subplots(1,3,figsize=(10,10))
axs[0].imshow(img_mpl[:,:,0],cmap="Reds")
axs[1].imshow(img_mpl[:,:,1],cmap="Greens")
axs[2].imshow(img_mpl[:,:,2],cmap="Blues")
axs[0].set_title("RED")
axs[1].set_title("GREEN")
axs[2].set_title("BLUE")
axs[0].axis('off')
axs[1].axis('off')
axs[2].axis('off')
plt.show()

fig , axs = plt.subplots(1,2,figsize=(10,10))
axs[0].imshow(img_mpl)
axs[0].axis('off')
axs[0].set_title('Matplotlib Image')
axs[1].imshow(img_cv)
axs[1].axis('off')
axs[1].set_title('CV Image')
plt.show()

img_cvrgb = cv2.cvtColor(img_cv,cv2.COLOR_BGR2RGB)
fig , ax = plt.subplots(figsize=(7,7))
ax.imshow(img_cvrgb)
ax.axis('off')
ax.set_title("Converted")
plt.show()

img = plt.imread(dog[0])
fig , ax =plt.subplots(figsize=(8,8))
ax.axis('off')
ax.imshow(img)
plt.show()

img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
fig , ax = plt.subplots(figsize=(5,5))
ax.axis('off')
ax.imshow(img_gray,cmap='Greys')
ax.set_title("Grey ")
plt.show()

img_resized = cv2.resize(img, None , fx=0.25 , fy=0.25) #pixelated image 0.25 is quarter of actual size
fig , ax = plt.subplots(figsize=(8,8))
ax.axis('off')
ax.set_title("Resized Image")
ax.imshow(img_resized)
plt.show()

img.shape
img_resized.shape

img_newsize=cv2.resize(img,(200,100))
fig , ax = plt.subplots(figsize=(5,5))
ax.axis('off')
ax.set_title("Based on width and height ")
ax.imshow(img_newsize)
plt.show()

kernel_sharpen=np.array([[-1,-1,-1],
                         [-1 , 9 , -1],
                         [-1,-1,-1]])
img_sharpened=cv2.filter2D(img,-1,kernel_sharpen)
fig , ax = plt.subplots(figsize=(8,8))
ax.axis('off')
ax.set_title("Sharpened Image")
ax.imshow(img_sharpened)
plt.show()

kernel_blur=np.array([[1/16,1/8,1/16],
                         [1/8 , 1/4 , 1/8],
                         [1/16,1/8,1/16]])
img_blur=cv2.filter2D(img,-1,kernel_blur)
fig , ax = plt.subplots(figsize=(8,8))
ax.axis('off')
ax.set_title("Blurred Image")
ax.imshow(img_blur)
plt.show()

plt.imsave('dog.png',img_blur)
cv2.imwrite('dog_cv2.png',img_blur)
