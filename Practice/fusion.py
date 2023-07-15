from skimage import io
import numpy as np
import cv2
imghigh = cv2.imread('G:/fenjie/23/high_highlight/1.png')
print(imghigh.max())
imglow = cv2.imread('G:/fenjie/23/low_highlight/1.png')
#imglow = cv2.imread('F:/Practice/146.jpg')
imglow=cv2.cvtColor(np.uint8(imglow),cv2.COLOR_BGR2RGB)
#img=cv2.cvtColor(np.uint8(imghigh),cv2.COLOR_BGR2GRAY)
img=np.minimum(imghigh/255+imglow/255,1)

cv2.imwrite("1234.jpg",img*255)
