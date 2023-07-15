from PIL import Image
import numpy as np

import cv2
file='G:/fenjie/15/high_lowlight/1.png'
img=cv2.imread(file)
img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
row, column = img.shape
img.astype("float")
Gauss_noise = np.random.normal(0,20, (row, column))
Gauss = img + Gauss_noise
img = np.where(Gauss < 0, 0, np.where(Gauss > 255, 255, Gauss))
cv2.imshow("pepper", img.astype("uint8"))
cv2.waitKey(5000)