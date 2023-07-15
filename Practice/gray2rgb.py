import numpy as np
import cv2
img=cv2.imread("G:/ceshi/LOL_DE_99/1.jpg")
img1=cv2.imread("G:/fenjie/14/low_lowlight/1.png")
hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
hsv[:,:,2]=img[:,:]
rgb=cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
cv2.imwrite("117.jpg",rgb)