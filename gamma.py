import cv2
import numpy as np
def gamma(image, gamma):
    invGamma = 1.0/gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** invGamma) * 255)
    table = np.array(table).astype("uint8")
    return cv2.LUT(image, table)
if __name__ == '__main__':
    img=cv2.imread("G:/NIKON_GT_V/21.jpg")
    img=gamma(img,2.2)
    cv2.imwrite("1.jpg",img)