import cv2
import numpy as np
import colortrans
import gamma


for i in  range(149,460):
    img=cv2.imread("H:/singleEi/GE/%d.png"%(i))
    balance1_image=colortrans.Raw2sRGB(img)
    image = gamma.gamma(balance1_image, 2.2)
    cv2.imwrite("H:/singleEi/GENEW/%d.png"%(i),image)
