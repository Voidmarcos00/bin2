import cv2 
import matplotlib.pyplot as plt
import numpy as np

image=cv2.imread("abhi.jpeg")
plt.imshow(image)
# cv2.imshow("123",image)

new = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
plt.imshow(new, cmap='Gray')  # Use cmap='gray' for grayscale images
plt.axis('off')  # Turn off axis
plt.show()
# cv2.imshow("123",new)


# 1.splitting image channel


r,g,b= cv2.split(new)
print('r',r.shape)
print('g',g.shape)
print('b',b.shape)

#resize of image

s=10
w=int(new.shape[1]*s/100)
h=int(new.shape[0]*s/100)
dim=(w,h)
resi=cv2.resize(new,dim,interpolation =cv2.INTER_AREA)
resi.shape

#rotate operation

(h,w) =new.shape[:2]
c= (w/2,h/2)
angle=90
m=cv2.getRotationMatrix2D(c,angle,1.0)
rotate_90 = cv2.warpAffine(new,m,(h,w))
plt.imshow(rotate_90, cmap='gray')  # Use cmap='gray' for grayscale images
plt.axis('on')  # Turn off axis
plt.show()
# cv2.imshow("123",rotate_90)