import cv2
import numpy as np
import os

name=os.listdir("F:/eyes/DATA/Normal")

for i in range(2000):
    img = cv2.imread(os.path.join("F:/eyes/DATA/Normal", name[i]))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("F:/eyes/DATA/NormalV/%d.jpg" % (int(i + 1)), hsv)

