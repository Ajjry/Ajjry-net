import cv2
import os
name=os.listdir("F:/eyes/data_R_L/normal/L")
name.sort(key=lambda x: int(x[:-4]))
for i in range(201):
    img = cv2.imread(os.path.join("F:/eyes/data_R_L/normal/L", name[i]))
    ret,thres1 = cv2.threshold(img,20, 200, cv2.THRESH_BINARY)
    cv2.imshow("1",thres1)
    cv2.waitKey(100)
