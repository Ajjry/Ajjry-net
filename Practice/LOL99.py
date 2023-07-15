import cv2
import os
import numpy as np

for i in range(1,16):
    img=cv2.imread("G:/fushion/%d.jpg"%(int(i)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #imghigh = cv2.imread('G:/fenjie/15/high_highlight/%d.png'%(int(i)))
    #imghigh = cv2.imread('G:/1/t/%d.jpg'%(int(i)))
    #imglow = img
    # img=cv2.cvtColor(np.uint8(imglow),cv2.COLOR_BGR2GRAY)
    # img=cv2.cvtColor(np.uint8(imghigh),cv2.COLOR_BGR2GRAY)
    #img = np.minimum(imghigh / 255 + imglow / 255, 1)

    cv2.imwrite("G:/fusion/%d.png"%(int(i)), img)