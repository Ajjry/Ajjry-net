import os

import cv2
import numpy as np


name_gt=os.listdir("G:/fusion/4/")
name_gt.sort(key=lambda x: int(x[:-4]))
for i in range(15):
    img=cv2.imread(os.path.join('G:/fusion/4/',name_gt[i]))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.imwrite('G:/fusion/4/%d.jpg'%(int(i+1)),img)