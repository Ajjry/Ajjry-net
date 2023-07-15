import cv2
import numpy as np
import os
name=os.listdir("F:/eyes/DATA/Normal")
for k in range(500):
    img = cv2.imread(os.path.join("F:/eyes/DATA/Normal", name[k]))
    img=img
    img_shape=img.shape
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            max=np.max(img[i,j,:])
            min=np.min(img[i,j,:])
            A=max/min
            if (A<1.31)&max!=0:
                img[i,j,:]=img[i,j,:]
            else:
                img[i,j,:]=[0,0,0]
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh1= cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
    ret,thresh2= cv2.threshold(img,230, 255, cv2.THRESH_BINARY)
    thresh3=thresh1-thresh2
    #out=cv2.GaussianBlur(thresh3,(5,5),1.3)
    cv2.imwrite("F:/eyes/DATA/BW_find/%d.jpg"%(k + 1), thresh3)
    ht= cv2.imread(os.path.join("F:/eyes/DATA/Normal", name[k]))
    cv2.imwrite("F:/eyes/DATA/BW_find/%d(go).jpg" % (k + 1), ht)