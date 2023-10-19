import cv2
import numpy as np
import math

img1=cv2.imread("C:/Users/admin/Desktop/r_low_l_high_decom_unet.png")
enhanceimg=cv2.imread("C:/Users/admin/Desktop/l_delta(1).png")
img_shape = enhanceimg.shape  # 图像大小(565, 650, 3)
print(img_shape)
h = img_shape[0]
w = img_shape[1]
# 彩色图像转换为灰度图像（3通道变为1通道）
gray = cv2.cvtColor(enhanceimg, cv2.COLOR_BGR2GRAY)
print(gray.shape)
# 最大图像灰度值减去原图像，即可得到反转的图像
dst = 255 - gray
dst1= np.expand_dims(dst, axis=2)
print(dst1.shape)
dst1= np.concatenate((dst1, dst1, dst1), axis=2)
img1=img1.astype(np.uint16)
dst1=dst1.astype(np.uint16)
img=np.minimum(img1+dst1,255)
img=img.astype(np.uint8)
#img = (img - np.min(img)) / (np.max(img) - np.min(img))
cv2.imshow("Lu",img)
cv2.waitKey(0)