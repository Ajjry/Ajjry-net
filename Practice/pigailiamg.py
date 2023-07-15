import cv2
import numpy as np
import os
name = os.listdir("G:/SONY_GT_V_decompose_S")
name.sort(key=lambda x: int(x[:-4]))
for k in range(650):

    img = cv2.imread(os.path.join("G:/NIKON_GT_V_decompose_S", name[k]))

    gamma = 4.4

    invGamma = 1.0 / gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** invGamma) * 255)
    table = np.array(table).astype("uint8")
    image = cv2.LUT(img, table)

    cv2.imwrite("G:/NIKON_highlight_V_decompose/%d.jpg"%(k + 1), image)
