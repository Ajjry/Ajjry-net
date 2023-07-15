import cv2
import numpy as np
import os

name=os.listdir("G:/fenjie/25/")
name.sort(key=lambda x: int(x[:-4]))
name_gt=os.listdir("G:/1/low/")
name_gt.sort(key=lambda x: int(x[:-4]))
for i in range(1000):
    img = cv2.imread(os.path.join("G:/fenjie/25/", name[i]))
    gt = cv2.imread(os.path.join("G:/1/low/", name_gt[i]))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gt=gt.astype(np.float)
    hsv=hsv.astype(np.float)
    tv=gt-hsv
    print(tv)
    tv=tv.astype(np.uint8)
    tv=tv
    tv = cv2.cvtColor(tv, cv2.COLOR_BGR2GRAY)

    cv2.imshow('1',tv)
    cv2.waitKey(5000)
    #cv2.imwrite("G:/fusion/%d.jpg" % (int(i + 1)),hsv)


