import cv2
import numpy as np

import os
name=os.listdir("F:/eyes/test/")
name.sort(key=lambda x: int(x[:-4]))
for i in range(1000):
        img = cv2.imread(os.path.join("F:/eyes/test/", name[i]))

        img=img[33:1033,300:1300]

        cv2.imwrite("F:/eyes/test/%d.jpg" % (int(i + 1)),img)