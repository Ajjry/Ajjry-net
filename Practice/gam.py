import cv2
import numpy as np
img=cv2.imread("G:/LOL_low_V_decompose/1.jpg")
gamma=4.4

invGamma = 1.0 / gamma
table = []
for i in range(256):
          table.append(((i / 255.0) ** invGamma) * 255)
table = np.array(table).astype("uint8")
image=cv2.LUT(img, table)
cv2.imwrite("2.jpg",image)
