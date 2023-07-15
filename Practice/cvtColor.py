import numpy as np
import cv2
img=cv2.imread("G:/fenjie/15/low_highlight/1.png")


img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


cv2.imwrite("146.jpg",img)